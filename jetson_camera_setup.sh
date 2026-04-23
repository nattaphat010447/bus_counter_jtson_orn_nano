#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Check if script is run as root (sudo)
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run this script with sudo: sudo ./jetson_camera_setup.sh"
  exit 1
fi

RULES_FILE="/etc/udev/rules.d/99-usb-cameras.rules"

echo "Creating udev rules at $RULES_FILE..."

# Write the rules to the file using cat << EOF
cat << 'EOF' > "$RULES_FILE"
# Top-Right USB Port
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.3:1.0", ATTR{index}=="0", SYMLINK+="top-right"

# Bottom-Right USB Port
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.4:1.0", ATTR{index}=="0", SYMLINK+="bottom-right"

# Top-Left USB Port
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.1:1.0", ATTR{index}=="0", SYMLINK+="top-left"

# Bottom-Left USB Port
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="platform-3610000.usb-usb-0:2.2:1.0", ATTR{index}=="0", SYMLINK+="bottom-left"
EOF

echo "Rules file created successfully."

echo "Reloading udev rules and triggering device manager..."
udevadm control --reload-rules
udevadm trigger --subsystem-match=video4linux

# Small pause to let the system create the symlinks if cameras are plugged in
sleep 2

echo "Setup complete!"
echo "--------------------------------------------------------"
echo "Currently connected and mapped cameras:"

# Check if any of the symlinks exist, suppress errors if they don't
ls -l /dev/top-* /dev/bottom-* 2>/dev/null || echo "   (No mapped cameras detected right now. Plug them in to see them!)"
echo "--------------------------------------------------------"