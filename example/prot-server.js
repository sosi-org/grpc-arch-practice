/*
Based on https://github.com/grpc/grpc/blob/v1.27.0/examples/node/dynamic_codegen/route_guide/route_guide_client.js
Also read:
https://techblog.fexcofts.com/2018/07/20/grpc-nodejs-chat-example/

*/

var async = require('async');

var grpc = require('grpc');
var protoLoader = require('@grpc/proto-loader');

var PROTO_PATH = __dirname + '/serv1.proto';

//Load protobuf
// Suggested options for similarity to existing grpc.load behavior
var packageDefinition = protoLoader.loadSync(
    PROTO_PATH,
    {
        keepCase: true,
        longs: String,
        enums: String,
        defaults: true,
        oneofs: true
    });
var protoPackageDef = grpc.loadPackageDefinition(packageDefinition);
//console.log(Object.keys(protoPackageDef)); :
// [ 'NumbersService', 'Number', 'NumStr' ]
console.log('>>', Object.keys(protoPackageDef.NumbersService));

// The protoPackageDef object has the full package hierarchy
var numbersService = protoPackageDef.NumbersService;
console.log('11', numbersService);
/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/


function toStr(num1) {
    // Why is num1 === 0 ?
    console.log('Server: ToStr:', num1);
    console.log('ToStr.arguments()', arguments);
    return '#' + num1 + '';
}

/**
 * getFeature request handler. Gets a request with a point, and responds with a
 * feature object indicating whether there is a feature at that point.
 * @param {EventEmitter} call Call object for the handler to process
 * @param {function(Error, feature)} callback Response callback
 */
function ToStr_SM(call, callback) {
    // call.next_call()
    console.log('server: call', call);
    console.log('server: call.request', call.request);
    callback(null, toStr(call.request));
}

function getServer() {
    var server = new grpc.Server();
    const s = protoPackageDef.NumbersService.service;
    //const s = numbersService.NumbersService.service;
    server.addProtoService(s, {
        ToStr: ToStr_SM,
      //listFeatures: listFeatures,
      //recordRoute: recordRoute,
      //routeChat: routeChat
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
