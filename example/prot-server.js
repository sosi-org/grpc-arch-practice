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
console.log('*>>>>>', NumbersService);  //  function ServiceClient(address, credentials, options)
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
 * ToStr request handler. Gets a request and responds ... .
 * @param {EventEmitter} callObj - The Call object for the handler to process
 * @param {function(Error, feature)} send_result_callback - The Response callback
 */
function ToStr_handler(callObj, send_result_callback) {
    // callObj.next_call()
    console.log('server: callObj received', callObj);
    console.log('server: callObj.request=', callObj.request);
    send_result_callback(null, toStr(callObj.request));
}

function newServer() {
    const server = new grpc.Server();
    const servc = NumbersService.service;
    server.addProtoService(servc, {
        ToStr: ToStr_handler,
      //listFeatures: listFeatures,
      //recordRoute: recordRoute,
      //routeChat: routeChat
    });
    return server;
}

const SERVING_ADDRESS = '0.0.0.0:50051';

function main() {
    // If this is run as a script, start a server on an unused port
    var routeServer = newServer();
    routeServer.bind(SERVING_ADDRESS, grpc.ServerCredentials.createInsecure());

    routeServer.start();

}

main();
