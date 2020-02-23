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
var protoPackageDef = grpc.loadPackageDefinition(packageDefinition);
// The protoPackageDef object has the full package hierarchy

console.log({protoPackageDef});

var NumbersService = protoPackageDef.NumbersService;
/*
console.log(protoPackageDef);
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

var clientStub = new protoPackageDef.NumbersService(SERVER_ADDRESS, grpc.credentials.createInsecure());

console.log('client: stub', clientStub);
console.log('client: stub: keys=', Object.keys(clientStub));
console.log('client stub ready.');
console.log();

//var client2 = new protoPackageDef.Number(SERVER_ADDRESS, grpc.credentials.createInsecure());
//console.log('client2', client2);


function demo1(finishedCalback) {
    // const p1 = 45;
    const p1 = { numval: 45 };
    // SOLVED! argument needs to be in the format specified by the .proto

    //console.log({clientStub})
    // despite 'ToStr' not seen as a key inside clientStub. Why??
    const call = clientStub.ToStr(p1, (error, result) => { /*received*/
        console.log('client: *return-callback*');
        if (error) {
            console.log('error detected:');
            finishedCalback(error);
            return;
        }
        // INCORRECT RESULT:
        console.log('grpc returned result:', result)
        // next()
        //console.log('next: starting');
        finishedCalback();
        // finishedCalback(); ---> Error: Callback was already called.
        //console.log('next: ok')
        console.log('client: done.')
    } );

    console.log('client: call sent: call object=', call);
    console.log();

    call.on('data', function(feature) {
        console.log('on data: Found stream data  "', feature);
    });

    // why?
    // call.on('end', finishedCalback);
    call.on('end', (...args) => {
        console.log('client: on "end" event:', args);
        return finishedCalback(null, 'best result333333333333333');
    });

    call.on('error', function(...args) {
        console.log('client: on error event:', args);
        console.log('        An error has occurred and the stream has been closed.');
    });

    call.on('status', function(...args) {
        console.log('client: process status event:');
        /*
        {
            code: 0,
            details: 'OK',
            metadata: Metadata { _internal_repr: {}, flags: 0 }
        }
        */
        console.log('        on status:', args);
    });

    //call.end('eee*');
    // return call;
}

async.series([
    demo1,
  ]);



/* output:

server: call.request { numval: 0 }
Server: ToStr: { numval: 0 }
ToStr.arguments() [Arguments] { '0': { numval: 0 } }
result: { strval: '' }
next: starting
next: ok
*/