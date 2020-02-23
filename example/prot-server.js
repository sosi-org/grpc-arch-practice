'use strict';

/*
Based on https://github.com/grpc/grpc/blob/v1.27.0/examples/node/dynamic_codegen/route_guide/route_guide_client.js
Also read:
https://techblog.fexcofts.com/2018/07/20/grpc-nodejs-chat-example/

*/

const async = require('async');
const grpc = require('grpc');


const require_protobuf = require('./require-protobuf.js');
const PROTO_PATH = __dirname + '/serv1.proto';

// require-like usage
const { NumbersService, Number, NumStr} = require_protobuf(PROTO_PATH);

console.log('>>', Object.keys(NumbersService));

// The protoPackageDef object has the full package hierarchy
const numbersService = NumbersService;

/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/


// num1 = { numval: * }
function toStr({numval:num1}) {
    // Why is num1 === 0 ?
    console.log('Server: ToStr:', num1); //  { numval: 0 }
    console.log('        ToStr.arguments()', arguments);
    // const result = '#' + num1 + '';
    const result = { strval: '#' + num1 + '' }

    console.log('        ToStr: result', result);
    return result;
}

/**
 * getFeature request handler. Gets a request with a point, and responds with a
 * feature object indicating whether there is a feature at that point.
 * @param {EventEmitter} call Call object for the handler to process
 * @param {function(Error, feature)} callback Response callback
 */
function ToStr_SM(call, callback) {
    // call.next_call()
    console.log('server: call received', call);
    console.log('server: call.request=', call.request);
    callback(null, toStr(call.request));
}

function getServer() {
    var server = new grpc.Server();
    const s = NumbersService.service;
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
