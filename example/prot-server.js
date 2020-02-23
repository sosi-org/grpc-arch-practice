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

function experimenting_with_schemas() {
    const chai = require('chai');
    chai.use(require('chai-json-schema'));


    var jsb = require('json-schema-builder');
    //var schema = jsb.schema();
    //const json = jsb.schema().json();
    //jsb.integer()
    const json_ = jsb; //.schema().json();
    console.log(json_);
    const intype = json_.object().property('numval', json_.number(), true);
    const outtype = json_.object().property('strval', json_.string(), true);
    console.log('intype', intype);
    console.log('outtype', outtype);
    //console.log('----', json_.format());

    const obj = {numval:2};
    chai.expect(obj).to.be.jsonSchema(intype);
    chai.expect(obj).to.be.jsonSchema(outtype);

    console.log();
    console.log();
}

// experimenting_with_schemas();

function experiments_2() {
    var mongoose = require('mongoose');
    var Schema = mongoose.Schema;
    const s = new Schema({
        numval:  Number
    });
    console.log('s', s);
}
// experiments_2 ();

function experiment_3() {
    const inptype =
    {
        "type": "map",
        "detail": [
            {
                "type": "number",
                "name": "numval"
            }
        ]
    };
    const outtype =
    {
        "type": "map",
        "detail": [
            {
                "type": "string",
                "name": "strval"
            }
        ]
    };
    // https://roger13.github.io/SwagDefGen/
    const t2 =
    {
        "numval": {
            "type": "integer",
            "format": "int32"
        },
        "required": ['numval'],
        //additionalProperties: false, //why fails?
    };
    const t3 =
    {
        "strval": {
            "type": "string",
            //required: true,
        },
        "required": ['strval'],
        //additionalProperties: false, //why fails?
    };
    const chai = require('chai');
    chai.use(require('chai-json-schema'));

    const obj = {numval: 2};
    const sobj = {strval: 'some text'};
    chai.expect(obj).to.be.jsonSchema(t2);
    chai.expect(obj).to.not.be.jsonSchema(t3);
    chai.expect(sobj).to.be.jsonSchema(t3);
    chai.expect(sobj).to.not.be.jsonSchema(t2);

    //chai.expect(obj).to.be.jsonSchema(inptype);
    //chai.expect(obj).to.be.jsonSchema(outtype);
    console.log('ok');
}

experiment_3();

// num1 = { numval: * }
function toStr({numval:num1}) {
    // Why is num1 === 0 ?
    console.log('Server: ToStr:', num1); //  { numval: 0 }
    console.log('        ToStr.arguments()', arguments);
    // const result = '#' + num1 + '';
    const strval = '#' + num1 + '';
    console.log('        ToStr: result', strval);
    return { strval: strval};
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
