/*
Based on https://github.com/grpc/grpc/blob/v1.27.0/examples/node/dynamic_codegen/route_guide/route_guide_client.js
*/

var async = require('async');

var PROTO_PATH = __dirname + '/serv1.proto';

var grpc = require('grpc');
var protoLoader = require('@grpc/proto-loader');
// Suggested options for similarity to existing grpc.load behavior
var packageDefinition = protoLoader.loadSync(
    PROTO_PATH,
    {keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true
    });
var protoDescriptor = grpc.loadPackageDefinition(packageDefinition);
//console.log(Object.keys(protoDescriptor)); :
// [ 'NumbersService', 'Number', 'NumStr' ]
console.log('>>', Object.keys(protoDescriptor.NumbersService));

// The protoDescriptor object has the full package hierarchy
var numbersService = protoDescriptor.NumbersService;
console.log('11', numbersService);
/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/

/*
var client = new numbersService.NumbersService(SERVER_ADDRESS, grpc.credentials.createInsecure());
console.log('client', client);
console.log('okk')
*/


function getServer() {
    var server = new grpc.Server();
    const s = protoDescriptor.NumbersService.service;
    //const s = numbersService.NumbersService.service;
    server.addProtoService(s, {
      getFeature: getFeature,
      listFeatures: listFeatures,
      recordRoute: recordRoute,
      routeChat: routeChat
    });
    return server;
}

const SERVING_ADDRESS = '0.0.0.0:50051';
function main() {
    // If this is run as a script, start a server on an unused port
    var routeServer = getServer();
    routeServer.bind(SERVING_ADDRESS, grpc.ServerCredentials.createInsecure());

    routeServer.start();

}

main();
