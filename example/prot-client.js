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

var numbersService = protoDescriptor.NumbersService;
/*
console.log(protoDescriptor);
console.log(numbersService);
*/

/*
NumbersService: { ToStr, GetNextNumbers, GenerateStrings, AddNumbers }
Number
NumStr
*/


const SERVER_ADDRESS = 'localhost:50051';

var client = new numbersService.NumbersService(SERVER_ADDRESS, grpc.credentials.createInsecure());

console.log('client', client);
console.log('okk')