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

class PolicyUpdate {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.obs = null;
      this.mu = null;
      this.prc = null;
      this.wt = null;
      this.dO = null;
      this.dPrimObs = null;
      this.dValObs = null;
      this.dU = null;
      this.n = null;
      this.rollout_len = null;
    }
    else {
      if (initObj.hasOwnProperty('obs')) {
        this.obs = initObj.obs
      }
      else {
        this.obs = [];
      }
      if (initObj.hasOwnProperty('mu')) {
        this.mu = initObj.mu
      }
      else {
        this.mu = [];
      }
      if (initObj.hasOwnProperty('prc')) {
        this.prc = initObj.prc
      }
      else {
        this.prc = [];
      }
      if (initObj.hasOwnProperty('wt')) {
        this.wt = initObj.wt
      }
      else {
        this.wt = [];
      }
      if (initObj.hasOwnProperty('dO')) {
        this.dO = initObj.dO
      }
      else {
        this.dO = 0;
      }
      if (initObj.hasOwnProperty('dPrimObs')) {
        this.dPrimObs = initObj.dPrimObs
      }
      else {
        this.dPrimObs = 0;
      }
      if (initObj.hasOwnProperty('dValObs')) {
        this.dValObs = initObj.dValObs
      }
      else {
        this.dValObs = 0;
      }
      if (initObj.hasOwnProperty('dU')) {
        this.dU = initObj.dU
      }
      else {
        this.dU = 0;
      }
      if (initObj.hasOwnProperty('n')) {
        this.n = initObj.n
      }
      else {
        this.n = 0;
      }
      if (initObj.hasOwnProperty('rollout_len')) {
        this.rollout_len = initObj.rollout_len
      }
      else {
        this.rollout_len = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PolicyUpdate
    // Serialize message field [obs]
    bufferOffset = _arraySerializer.float32(obj.obs, buffer, bufferOffset, null);
    // Serialize message field [mu]
    bufferOffset = _arraySerializer.float32(obj.mu, buffer, bufferOffset, null);
    // Serialize message field [prc]
    bufferOffset = _arraySerializer.float32(obj.prc, buffer, bufferOffset, null);
    // Serialize message field [wt]
    bufferOffset = _arraySerializer.float32(obj.wt, buffer, bufferOffset, null);
    // Serialize message field [dO]
    bufferOffset = _serializer.int32(obj.dO, buffer, bufferOffset);
    // Serialize message field [dPrimObs]
    bufferOffset = _serializer.int32(obj.dPrimObs, buffer, bufferOffset);
    // Serialize message field [dValObs]
    bufferOffset = _serializer.int32(obj.dValObs, buffer, bufferOffset);
    // Serialize message field [dU]
    bufferOffset = _serializer.int32(obj.dU, buffer, bufferOffset);
    // Serialize message field [n]
    bufferOffset = _serializer.int32(obj.n, buffer, bufferOffset);
    // Serialize message field [rollout_len]
    bufferOffset = _serializer.int32(obj.rollout_len, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PolicyUpdate
    let len;
    let data = new PolicyUpdate(null);
    // Deserialize message field [obs]
    data.obs = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [mu]
    data.mu = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [prc]
    data.prc = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [wt]
    data.wt = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [dO]
    data.dO = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [dPrimObs]
    data.dPrimObs = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [dValObs]
    data.dValObs = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [dU]
    data.dU = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [n]
    data.n = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [rollout_len]
    data.rollout_len = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.obs.length;
    length += 4 * object.mu.length;
    length += 4 * object.prc.length;
    length += 4 * object.wt.length;
    return length + 40;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/PolicyUpdate';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1688550284fb9359a8dfabdfd917a70f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] obs
    float32[] mu
    float32[] prc
    float32[] wt
    
    int32 dO
    int32 dPrimObs
    int32 dValObs
    int32 dU
    int32 n
    int32 rollout_len
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PolicyUpdate(null);
    if (msg.obs !== undefined) {
      resolved.obs = msg.obs;
    }
    else {
      resolved.obs = []
    }

    if (msg.mu !== undefined) {
      resolved.mu = msg.mu;
    }
    else {
      resolved.mu = []
    }

    if (msg.prc !== undefined) {
      resolved.prc = msg.prc;
    }
    else {
      resolved.prc = []
    }

    if (msg.wt !== undefined) {
      resolved.wt = msg.wt;
    }
    else {
      resolved.wt = []
    }

    if (msg.dO !== undefined) {
      resolved.dO = msg.dO;
    }
    else {
      resolved.dO = 0
    }

    if (msg.dPrimObs !== undefined) {
      resolved.dPrimObs = msg.dPrimObs;
    }
    else {
      resolved.dPrimObs = 0
    }

    if (msg.dValObs !== undefined) {
      resolved.dValObs = msg.dValObs;
    }
    else {
      resolved.dValObs = 0
    }

    if (msg.dU !== undefined) {
      resolved.dU = msg.dU;
    }
    else {
      resolved.dU = 0
    }

    if (msg.n !== undefined) {
      resolved.n = msg.n;
    }
    else {
      resolved.n = 0
    }

    if (msg.rollout_len !== undefined) {
      resolved.rollout_len = msg.rollout_len;
    }
    else {
      resolved.rollout_len = 0
    }

    return resolved;
    }
};

module.exports = PolicyUpdate;
