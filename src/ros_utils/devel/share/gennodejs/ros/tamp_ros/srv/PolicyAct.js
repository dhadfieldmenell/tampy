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

class PolicyActRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.obs = null;
      this.noise = null;
      this.task = null;
    }
    else {
      if (initObj.hasOwnProperty('obs')) {
        this.obs = initObj.obs
      }
      else {
        this.obs = [];
      }
      if (initObj.hasOwnProperty('noise')) {
        this.noise = initObj.noise
      }
      else {
        this.noise = [];
      }
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PolicyActRequest
    // Serialize message field [obs]
    bufferOffset = _arraySerializer.float32(obj.obs, buffer, bufferOffset, null);
    // Serialize message field [noise]
    bufferOffset = _arraySerializer.float32(obj.noise, buffer, bufferOffset, null);
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PolicyActRequest
    let len;
    let data = new PolicyActRequest(null);
    // Deserialize message field [obs]
    data.obs = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [noise]
    data.noise = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.obs.length;
    length += 4 * object.noise.length;
    length += object.task.length;
    return length + 12;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PolicyActRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5ac251624ebf069341d6a35799d6d1a6';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] obs
    float32[] noise
    string task
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PolicyActRequest(null);
    if (msg.obs !== undefined) {
      resolved.obs = msg.obs;
    }
    else {
      resolved.obs = []
    }

    if (msg.noise !== undefined) {
      resolved.noise = msg.noise;
    }
    else {
      resolved.noise = []
    }

    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    return resolved;
    }
};

class PolicyActResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.act = null;
    }
    else {
      if (initObj.hasOwnProperty('act')) {
        this.act = initObj.act
      }
      else {
        this.act = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PolicyActResponse
    // Serialize message field [act]
    bufferOffset = _arraySerializer.float32(obj.act, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PolicyActResponse
    let len;
    let data = new PolicyActResponse(null);
    // Deserialize message field [act]
    data.act = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.act.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PolicyActResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '90891bbf8bb50ef502f4296a9265197f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    float32[] act
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PolicyActResponse(null);
    if (msg.act !== undefined) {
      resolved.act = msg.act;
    }
    else {
      resolved.act = []
    }

    return resolved;
    }
};

module.exports = {
  Request: PolicyActRequest,
  Response: PolicyActResponse,
  md5sum() { return 'e3eb5859ffc1c0de9d569f656c4594dc'; },
  datatype() { return 'tamp_ros/PolicyAct'; }
};
