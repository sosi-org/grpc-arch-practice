'use strict';

/*
Usage:
    PNAME = './schema-validator-bind'
    require(PNAME)(NumStr)
    require(PNAME)(Number)
*/

const chai = require('chai');

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

//experiment_3();

const PROTO_PATH = __dirname + '/serv1.proto';

function experiment_4() {
  const fs = require('fs');
  const schema = require('protocol-buffers-schema');
  var sch = schema.parse(fs.readFileSync(PROTO_PATH))
  //schema.parse(protobufSchemaBufferOrString)

  //console.log('...',schema.stringify(sch));
  console.log('........................................SCHEMA:');
  console.log(sch);

  /*
  // Trying to extract the schema directly frmo the .proto file:

  const t2 = sch.messages.
  TODO: convert sch to swagger-style schema.

  */

  /*

  chai.use(require('chai-json-schema'));
  const obj = {numval: 2};
  const sobj = {strval: 'some text'};
  chai.expect(obj).to.be.jsonSchema(t2);
  chai.expect(obj).to.not.be.jsonSchema(t3);
  */

}
/*
experiment_4()
console.log('exiting');
process.exit(0);
*/

function experiment_5() {
  const chai = require('chai');
  chai.expect({numval:4}).to.have.nested.property('numval');
  chai.expect({strval:"df"}).to.have.nested.property('strval');
}
experiment_5();


/*
    //expect(callObj.request).to.have.nested.property('numval');

    chai.expect(callObj.request).to.have.nested.property('numval');

    chai.expect(resultObj).to.have.nested.property('strval');
*/

function bind_validation(class_name) {
  const chai = require('chai');
  function validator1(obj) {
    chai.expect(obj).to.have.nested.property('numval');
  }
  function validator2(obj) {
    chai.expect(obj).to.have.nested.property('strval');
  }
  class_name.validate = validator1;
}

module.exports = bind_validation;
