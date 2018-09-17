// Auto-generated. Do not edit!

// (in-package tamp_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let FloatArray = require('./FloatArray.js');

//-----------------------------------------------------------

class PlanResult {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.prob_id = null;
      this.trajectory = null;
      this.success = null;
      this.failed_preds = null;
    }
    else {
      if (initObj.hasOwnProperty('prob_id')) {
        this.prob_id = initObj.prob_id
      }
      else {
        this.prob_id = 0;
      }
      if (initObj.hasOwnProperty('trajectory')) {
        this.trajectory = initObj.trajectory
      }
      else {
        this.trajectory = [];
      }
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('failed_preds')) {
        this.failed_preds = initObj.failed_preds
      }
      else {
        this.failed_preds = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PlanResult
    // Serialize message field [prob_id]
    bufferOffset = _serializer.int64(obj.prob_id, buffer, bufferOffset);
    // Serialize message field [trajectory]
    // Serialize the length for message field [trajectory]
    bufferOffset = _serializer.uint32(obj.trajectory.length, buffer, bufferOffset);
    obj.trajectory.forEach((val) => {
      bufferOffset = FloatArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [failed_preds]
    bufferOffset = _arraySerializer.string(obj.failed_preds, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PlanResult
    let len;
    let data = new PlanResult(null);
    // Deserialize message field [prob_id]
    data.prob_id = _deserializer.int64(buffer, bufferOffset);
    // Deserialize message field [trajectory]
    // Deserialize array length for message field [trajectory]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.trajectory = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.trajectory[i] = FloatArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [failed_preds]
    data.failed_preds = _arrayDeserializer.string(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.trajectory.forEach((val) => {
      length += FloatArray.getMessageSize(val);
    });
    object.failed_preds.forEach((val) => {
      length += 4 + val.length;
    });
    return length + 17;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/PlanResult';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'd4f5f1c50852a30db764ffda62f46133';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int64 prob_id
    FloatArray[] trajectory
    bool success
    string[] failed_preds
    
    
    ================================================================================
    MSG: tamp_ros/FloatArray
    float32[] data
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PlanResult(null);
    if (msg.prob_id !== undefined) {
      resolved.prob_id = msg.prob_id;
    }
    else {
      resolved.prob_id = 0
    }

    if (msg.trajectory !== undefined) {
      resolved.trajectory = new Array(msg.trajectory.length);
      for (let i = 0; i < resolved.trajectory.length; ++i) {
        resolved.trajectory[i] = FloatArray.Resolve(msg.trajectory[i]);
      }
    }
    else {
      resolved.trajectory = []
    }

    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.failed_preds !== undefined) {
      resolved.failed_preds = msg.failed_preds;
    }
    else {
      resolved.failed_preds = []
    }

    return resolved;
    }
};

module.exports = PlanResult;
