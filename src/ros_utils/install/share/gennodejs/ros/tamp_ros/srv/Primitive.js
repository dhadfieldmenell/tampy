// Auto-generated. Do not edit!

// (in-package tamp_ros.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------


//-----------------------------------------------------------

class PrimitiveRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.prim_obs = null;
    }
    else {
      if (initObj.hasOwnProperty('prim_obs')) {
        this.prim_obs = initObj.prim_obs
      }
      else {
        this.prim_obs = 0.0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PrimitiveRequest
    // Serialize message field [prim_obs]
    bufferOffset = _serializer.float32(obj.prim_obs, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PrimitiveRequest
    let len;
    let data = new PrimitiveRequest(null);
    // Deserialize message field [prim_obs]
    data.prim_obs = _deserializer.float32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PrimitiveRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1014880aa631514a1032e57a819edaa3';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32 prim_obs
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PrimitiveRequest(null);
    if (msg.prim_obs !== undefined) {
      resolved.prim_obs = msg.prim_obs;
    }
    else {
      resolved.prim_obs = 0.0
    }

    return resolved;
    }
};

class PrimitiveResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.task_distr = null;
      this.obj_distr = null;
      this.targ_distr = null;
    }
    else {
      if (initObj.hasOwnProperty('task_distr')) {
        this.task_distr = initObj.task_distr
      }
      else {
        this.task_distr = [];
      }
      if (initObj.hasOwnProperty('obj_distr')) {
        this.obj_distr = initObj.obj_distr
      }
      else {
        this.obj_distr = [];
      }
      if (initObj.hasOwnProperty('targ_distr')) {
        this.targ_distr = initObj.targ_distr
      }
      else {
        this.targ_distr = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PrimitiveResponse
    // Serialize message field [task_distr]
    bufferOffset = _arraySerializer.float32(obj.task_distr, buffer, bufferOffset, null);
    // Serialize message field [obj_distr]
    bufferOffset = _arraySerializer.float32(obj.obj_distr, buffer, bufferOffset, null);
    // Serialize message field [targ_distr]
    bufferOffset = _arraySerializer.float32(obj.targ_distr, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PrimitiveResponse
    let len;
    let data = new PrimitiveResponse(null);
    // Deserialize message field [task_distr]
    data.task_distr = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [obj_distr]
    data.obj_distr = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [targ_distr]
    data.targ_distr = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.task_distr.length;
    length += 4 * object.obj_distr.length;
    length += 4 * object.targ_distr.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PrimitiveResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'e7c5b74c5540db8867d8b5db7fc110e8';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    float32[] task_distr
    float32[] obj_distr
    float32[] targ_distr
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PrimitiveResponse(null);
    if (msg.task_distr !== undefined) {
      resolved.task_distr = msg.task_distr;
    }
    else {
      resolved.task_distr = []
    }

    if (msg.obj_distr !== undefined) {
      resolved.obj_distr = msg.obj_distr;
    }
    else {
      resolved.obj_distr = []
    }

    if (msg.targ_distr !== undefined) {
      resolved.targ_distr = msg.targ_distr;
    }
    else {
      resolved.targ_distr = []
    }

    return resolved;
    }
};

module.exports = {
  Request: PrimitiveRequest,
  Response: PrimitiveResponse,
  md5sum() { return 'ec8948c09b640bcf5ec37fe64f2d51b1'; },
  datatype() { return 'tamp_ros/Primitive'; }
};
