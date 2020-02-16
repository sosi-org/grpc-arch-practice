/*
Based on https://github.com/grpc/grpc/blob/v1.27.0/examples/node/dynamic_codegen/route_guide/route_guide_client.js
*/

var async = require('async');

var PROTO_PATH = //__dirname + '/../../../protos/route_guide.proto';
	__dirname + '/serv1.proto';

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
// The protoDescriptor object has the full package hierarchy

console.log({protoDescriptor});

var NumbersService = protoDescriptor.NumbersService;
/*
console.log(protoDescriptor);
console.log(numbersService);
*/

console.log('----')
console.log('numbersService::::', NumbersService);
/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/


const SERVER_ADDRESS = 'localhost:50051';

var clientStub = new protoDescriptor.NumbersService(SERVER_ADDRESS, grpc.credentials.createInsecure());

console.log('client', clientStub);
console.log('okk');

//var client2 = new protoDescriptor.Number(SERVER_ADDRESS, grpc.credentials.createInsecure());
//console.log('client2', client2);


function demo1(finishedCalback) {
    const p1 = 45;
    //console.log({clientStub})
    // despite 'ToStr' not inside clientStub.
    const call = clientStub.ToStr(p1, (error, result) => {
        if (error) {
            console.log('error detected:');
            finishedCalback(error);
            return;
        }
        // INCORRECT RESULT:
        console.log('result:', result)
        // next()
        console.log('next: starting')
        finishedCalback();
        // finishedCalback(); ---> Error: Callback was already called.
        console.log('next: ok')
    } );

    console.log('q:call', call);
/*
    call.on('data', function(feature) {
        console.log('Found stream data  "', feature);
    });
    call.on('end', finishedCalback);
*/
}

async.series([
    demo1,
  ]);