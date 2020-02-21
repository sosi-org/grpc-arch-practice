set -ex
node prot-server.js &
server_pid=$1
echo server pid: $server_pid
echo "sleep 1"
sleep 1
echo ""
node prot-client.js
# wait $server_pid
wait %1

grc ps aux |grep "node prot"
