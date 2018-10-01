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

class QValueRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.obs = null;
    }
    else {
      if (initObj.hasOwnProperty('obs')) {
        this.obs = initObj.obs
      }
      else {
        this.obs = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QValueRequest
    // Serialize message field [obs]
    bufferOffset = _arraySerializer.float32(obj.obs, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QValueRequest
    let len;
    let data = new QValueRequest(null);
    // Deserialize message field [obs]
    data.obs = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.obs.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/QValueRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '78db2b7101dd13e67ac3a3430de0510d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] obs
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QValueRequest(null);
    if (msg.obs !== undefined) {
      resolved.obs = msg.obs;
    }
    else {
      resolved.obs = []
    }

    return resolved;
    }
};

class QValueResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.value = null;
    }
    else {
      if (initObj.hasOwnProperty('value')) {
        this.value = initObj.value
      }
      else {
        this.value = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type QValueResponse
    // Serialize message field [value]
    bufferOffset = _arraySerializer.float32(obj.value, buffer, bufferOffset, null);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type QValueResponse
    let len;
    let data = new QValueResponse(null);
    // Deserialize message field [value]
    data.value = _arrayDeserializer.float32(buffer, bufferOffset, null)
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.value.length;
    return length + 4;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/QValueResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '1becc0cb8362a822e3753aa6cf42cf70';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    float32[] value
    
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new QValueResponse(null);
    if (msg.value !== undefined) {
      resolved.value = msg.value;
    }
    else {
      resolved.value = []
    }

    return resolved;
    }
};

module.exports = {
  Request: QValueRequest,
  Response: QValueResponse,
  md5sum() { return '69d0eb61126056c55900069d173c5835'; },
  datatype() { return 'tamp_ros/QValue'; }
};
