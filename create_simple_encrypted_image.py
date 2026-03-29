#!/usr/bin/env python3
"""
Create a simple encrypted APFS image with a weak password for brute force testing.
"""

import os
import subprocess
import sys
import hashlib
import json

def create_encrypted_image(output_path, password, size_mb=100):
    """Create an encrypted APFS image with test files."""
    
    # Remove existing image if present
    if os.path.exists(output_path):
        os.remove(output_path)
    
    print(f"Creating {size_mb}MB encrypted APFS image...")
    print(f"Password: '{password}' (for testing only!)")
    
    # Create DMG
    print("\nStep 1: Creating DMG...")
    subprocess.run([
        'hdiutil', 'create',
        '-size', f'{size_mb}m',
        '-fs', 'APFS',
        '-volname', 'TestVolume',
        output_path
    ], check=True)
    
    # Attach and get device
    print("\nStep 2: Attaching DMG...")
    result = subprocess.run([
        'hdiutil', 'attach', output_path
    ], capture_output=True, text=True, check=True)
    
    # Parse hdiutil output to find device and mount point
    device = None
    mount_point = None
    
    for line in result.stdout.split('\n'):
        if '/dev/disk' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith('/dev/disk'):
                    device = part
                    # Mount point is usually the last part
                    if i + 1 < len(parts):
                        potential_mount = parts[-1]
                        if '/Volumes/' in potential_mount:
                            mount_point = potential_mount
                    break
            if device:
                break
    
    # If mount point not found, try to find it
    if not mount_point:
        result2 = subprocess.run(['diskutil', 'list'], capture_output=True, text=True)
        for line in result2.stdout.split('\n'):
            if 'TestVolume' in line:
                # Try to get mount point from diskutil
                if device:
                    device_id = device.replace('/dev/', '')
                    result3 = subprocess.run(['diskutil', 'info', device_id], 
                                            capture_output=True, text=True)
                    for info_line in result3.stdout.split('\n'):
                        if 'Mount Point:' in info_line:
                            mount_point = info_line.split(':')[1].strip()
                            break
    
    if not device:
        raise RuntimeError("Could not find device")
    
    if not mount_point:
        mount_point = '/Volumes/TestVolume'
    
    print(f"  Device: {device}")
    print(f"  Mount point: {mount_point}")
    
    # Encrypt volume
    print("\nStep 3: Encrypting volume...")
    # Find the APFS volume identifier
    device_id = device.replace('/dev/', '')
    
    # Get info about the mount point to find the volume
    result_info = subprocess.run(['diskutil', 'info', mount_point], 
                                capture_output=True, text=True)
    
    volume_id = None
    for line in result_info.stdout.split('\n'):
        if 'Device Node:' in line:
            volume_id = line.split(':')[1].strip().replace('/dev/', '')
            break
    
    if not volume_id:
        # Fallback: try to find from apfs list
        result_list = subprocess.run(['diskutil', 'apfs', 'list'], 
                                    capture_output=True, text=True)
        for line in result_list.stdout.split('\n'):
            if 'TestVolume' in line:
                parts = line.split()
                for part in parts:
                    if part.startswith('disk') and 's' in part:
                        volume_id = part
                        break
                if volume_id:
                    break
    
    if not volume_id:
        volume_id = device_id
    
    print(f"  Volume ID: {volume_id}")
    
    # Try encryption with -user disk first
    result = subprocess.run([
        'diskutil', 'apfs', 'encryptVolume', f'/dev/{volume_id}',
        '-user', 'disk',
        '-passphrase', password
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        # Try with full device path and -user disk
        print("  Trying with full device path...")
        result = subprocess.run([
            'diskutil', 'apfs', 'encryptVolume', f'/dev/{volume_id}',
            '-user', 'disk',
            '-passphrase', password
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            # Try creating disk user first
            print("  Creating disk user...")
            subprocess.run([
                'diskutil', 'apfs', 'addUser', f'/dev/{volume_id}',
                '-user', 'disk',
                '-passphrase', password
            ], capture_output=True, text=True)
            
            # Then encrypt
            result = subprocess.run([
                'diskutil', 'apfs', 'encryptVolume', f'/dev/{volume_id}',
                '-user', 'disk',
                '-passphrase', password
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                # Check if already encrypted
                if "already FileVaulted" in result.stderr or "already encrypted" in result.stderr:
                    print("  ✓ Volume is already encrypted")
                else:
                    print(f"  ERROR: Encryption failed: {result.stderr}")
                    print(f"  Output: {result.stdout}")
                    raise RuntimeError("Failed to encrypt volume")
    
    # Wait for encryption to complete
    import time
    print("  Waiting for encryption to complete...")
    time.sleep(5)
    
    # Create test files
    print("\nStep 4: Creating test files...")
    test_files = [
        ('test1.txt', 'This is test file 1'),
        ('test2.txt', 'This is test file 2'),
        ('test3.txt', 'This is test file 3'),
        ('data.json', '{"test": "data", "number": 42}'),
        ('readme.md', '# Test Volume\n\nThis is a test encrypted volume.'),
    ]
    
    for filename, content in test_files:
        filepath = os.path.join(mount_point, filename)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Created: {filename}")
    
    # Create a larger file
    large_file = os.path.join(mount_point, 'large.dat')
    with open(large_file, 'wb') as f:
        f.write(b'X' * (10 * 1024 * 1024))  # 10MB
    print(f"  Created: large.dat (10MB)")
    
    # Calculate file hashes
    print("\nStep 5: Calculating file hashes...")
    hashes = {}
    for filename, _ in test_files:
        filepath = os.path.join(mount_point, filename)
        with open(filepath, 'rb') as f:
            hashes[filename] = hashlib.sha256(f.read()).hexdigest()
    
    # Hash large file
    with open(large_file, 'rb') as f:
        hashes['large.dat'] = hashlib.sha256(f.read()).hexdigest()
    
    # Save hashes
    hash_file = output_path.replace('.dmg', '_hashes.json')
    with open(hash_file, 'w') as f:
        json.dump(hashes, f, indent=2)
    print(f"  Hashes saved to: {hash_file}")
    
    # Detach
    print("\nStep 6: Detaching DMG...")
    subprocess.run(['hdiutil', 'detach', device, '-force'], check=True)
    
    print(f"\n✓ Encrypted image created: {output_path}")
    print(f"✓ Password: '{password}'")
    print(f"✓ Hash file: {hash_file}")
    
    return output_path, hash_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_simple_encrypted_image.py <output.dmg> [password] [size_mb]")
        print("Example: python create_simple_encrypted_image.py test_weak.dmg ab 100")
        sys.exit(1)
    
    output_path = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) > 2 else 'ab'
    size_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    
    create_encrypted_image(output_path, password, size_mb)

if __name__ == '__main__':
    main()

