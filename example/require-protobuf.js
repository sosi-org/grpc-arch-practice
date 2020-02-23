'use strict';

const grpc = require('grpc');

/*
Load protobuf: dynamic version
proto-require()!

@returns:
    The protoPackageDef object has the full package hierarchy
    { NumbersService, Number, NumStr}

Usage:
    const { NumbersService, Number, NumStr} = require_protobuf(PROTO_PATH);
*/
function require_protobuf(proto_file_path) {

    const protoLoader = require('@grpc/proto-loader');

    // Suggested options for similarity to existing grpc.load behavior
    const packageDefinition = protoLoader.loadSync(
        proto_file_path,
        {
            keepCase: true,
            longs: String,
            enums: String,
            defaults: true,
            oneofs: true
        });
    const protoPackageDef = grpc.loadPackageDefinition(packageDefinition);
    // The protoPackageDef object has the full package hierarchy
    // { NumbersService, Number, NumStr}
    return protoPackageDef;
}

module.exports = require_protobuf;
