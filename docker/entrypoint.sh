#!/bin/bash

echo "Entering entrypoint"

echo "Enabling user to SSH"

grep -qxF "${USERNAME}:x:$(id -u):0::${HOME}:/bin/bash" /etc/passwd || echo "${USERNAME}:x:$(id -u):0::${HOME}:/bin/bash" >> /etc/passwd
grep -qxF "${USERNAME}:x:$(id -u):" /etc/group || echo "${USERNAME}:x:$(id -u):" >> /etc/group

chmod 755 ${HOME}

echo "Starting SSH"

dropbear -wse -p 0.0.0.0:2222 -r /dropbear/id_dropbear

echo "Done"

if [[ $# -eq 0 ]]; then
  exec "/bin/bash -l"
else
  exec "$@"
fi
