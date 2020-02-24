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

console.log('>>', Object.keys(NumbersService)); //[ 'super_', 'service' ]
console.log('*>>>>>', NumbersService);  //  function ServiceClient(address, credentials, options)
/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/

console.log('---', NumStr);

// not functional yet
const x__ = require('./schema-validation-practice')(NumStr);

// num1 = { numval: * }
function toStr({numval:num1}) {
    // Why is num1 === 0 ?
    chai.expect(num1).to.not.be.undefined;
    console.log('Server: ToStr:', num1); //  { numval: 0 }
    console.log('        ToStr.arguments()', arguments);
    // const result = '#' + num1 + '';
    const strval = '#' + num1 + '';
    console.log('        ToStr: result', strval);
    return { strval: strval};
}

const chai = require('chai');

/**
 * ToStr request handler. Gets a request and responds ... .
 * @param {EventEmitter} callObj - The Call object for the handler to process
 * @param {function(Error, feature)} send_result_callback - The Response callback
 */
function ToStr_handler(callObj, send_result_callback) {
    // callObj.next_call()
    console.log('serv**er: callObj received', callObj);
    /*
    callObj.metadata = Metadata {  _internal_repr: { 'user-agent': [ 'grpc-node/1.24.2 grpc-c/8.0.0 (linux; chttp2; ganges)' ] }, flags: 0 }
    */
    console.log('server: callObj.request=', callObj.request);
    //expect(callObj.request).to.have.nested.property('numval');

    chai.expect(callObj.request).to.have.nested.property('numval');

    const resultObj = toStr(callObj.request);

    chai.expect(resultObj).to.have.nested.property('strval');

    send_result_callback(null, resultObj);
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
