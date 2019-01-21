// Auto-generated. Do not edit!

// (in-package tamp_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class HLProblem {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.solver_id = null;
      this.server_id = null;
      this.init_state = null;
      this.cond = null;
      this.path_to = null;
      this.gmms = null;
    }
    else {
      if (initObj.hasOwnProperty('solver_id')) {
        this.solver_id = initObj.solver_id
      }
      else {
        this.solver_id = 0;
      }
      if (initObj.hasOwnProperty('server_id')) {
        this.server_id = initObj.server_id
      }
      else {
        this.server_id = 0;
      }
      if (initObj.hasOwnProperty('init_state')) {
        this.init_state = initObj.init_state
      }
      else {
        this.init_state = [];
      }
      if (initObj.hasOwnProperty('cond')) {
        this.cond = initObj.cond
      }
      else {
        this.cond = 0;
      }
      if (initObj.hasOwnProperty('path_to')) {
        this.path_to = initObj.path_to
      }
      else {
        this.path_to = '';
      }
      if (initObj.hasOwnProperty('gmms')) {
        this.gmms = initObj.gmms
      }
      else {
        this.gmms = '';
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type HLProblem
    // Serialize message field [solver_id]
    bufferOffset = _serializer.int32(obj.solver_id, buffer, bufferOffset);
    // Serialize message field [server_id]
    bufferOffset = _serializer.int32(obj.server_id, buffer, bufferOffset);
    // Serialize message field [init_state]
    bufferOffset = _arraySerializer.float32(obj.init_state, buffer, bufferOffset, null);
    // Serialize message field [cond]
    bufferOffset = _serializer.int32(obj.cond, buffer, bufferOffset);
    // Serialize message field [path_to]
    bufferOffset = _serializer.string(obj.path_to, buffer, bufferOffset);
    // Serialize message field [gmms]
    bufferOffset = _serializer.string(obj.gmms, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type HLProblem
    let len;
    let data = new HLProblem(null);
    // Deserialize message field [solver_id]
    data.solver_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [server_id]
    data.server_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [init_state]
    data.init_state = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [cond]
    data.cond = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [path_to]
    data.path_to = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [gmms]
    data.gmms = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.init_state.length;
    length += object.path_to.length;
    length += object.gmms.length;
    return length + 24;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/HLProblem';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '573d764c14495d91c035cf75dd9b1437';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 solver_id
    int32 server_id
    float32[] init_state
    int32 cond
    string path_to
    string gmms
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new HLProblem(null);
    if (msg.solver_id !== undefined) {
      resolved.solver_id = msg.solver_id;
    }
    else {
      resolved.solver_id = 0
    }

    if (msg.server_id !== undefined) {
      resolved.server_id = msg.server_id;
    }
    else {
      resolved.server_id = 0
    }

    if (msg.init_state !== undefined) {
      resolved.init_state = msg.init_state;
    }
    else {
      resolved.init_state = []
    }

    if (msg.cond !== undefined) {
      resolved.cond = msg.cond;
    }
    else {
      resolved.cond = 0
    }

    if (msg.path_to !== undefined) {
      resolved.path_to = msg.path_to;
    }
    else {
      resolved.path_to = ''
    }

    if (msg.gmms !== undefined) {
      resolved.gmms = msg.gmms;
    }
    else {
      resolved.gmms = ''
    }

    return resolved;
    }
};

module.exports = HLProblem;
