set -ex
node prot-server.js &
server_pid=%1
echo "skeep 1"
sleep 1
node prot-client.js
# wait $server_pid
# fg

