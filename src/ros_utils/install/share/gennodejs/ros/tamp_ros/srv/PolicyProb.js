// Auto-generated. Do not edit!

// (in-package tamp_ros.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class PolicyProbRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.obs = null;
      this.task = null;
    }
    else {
      if (initObj.hasOwnProperty('obs')) {
        this.obs = initObj.obs
      }
      else {
        this.obs = [];
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
    // Serializes a message object of type PolicyProbRequest
    // Serialize message field [obs]
    // Serialize the length for message field [obs]
    bufferOffset = _serializer.uint32(obj.obs.length, buffer, bufferOffset);
    obj.obs.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PolicyProbRequest
    let len;
    let data = new PolicyProbRequest(null);
    // Deserialize message field [obs]
    // Deserialize array length for message field [obs]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.obs = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.obs[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.obs.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    length += object.task.length;
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PolicyProbRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '2d1d9d4710e5ea79eaea0079446cb151';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    std_msgs/Float32MultiArray[] obs
    string task
    
    
    ================================================================================
    MSG: std_msgs/Float32MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    float32[]         data          # array of data
    
    
    ================================================================================
    MSG: std_msgs/MultiArrayLayout
    # The multiarray declares a generic multi-dimensional array of a
    # particular data type.  Dimensions are ordered from outer most
    # to inner most.
    
    MultiArrayDimension[] dim # Array of dimension properties
    uint32 data_offset        # padding elements at front of data
    
    # Accessors should ALWAYS be written in terms of dimension stride
    # and specified outer-most dimension first.
    # 
    # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
    #
    # A standard, 3-channel 640x480 image with interleaved color channels
    # would be specified as:
    #
    # dim[0].label  = "height"
    # dim[0].size   = 480
    # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
    # dim[1].label  = "width"
    # dim[1].size   = 640
    # dim[1].stride = 3*640 = 1920
    # dim[2].label  = "channel"
    # dim[2].size   = 3
    # dim[2].stride = 3
    #
    # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
    
    ================================================================================
    MSG: std_msgs/MultiArrayDimension
    string label   # label of given dimension
    uint32 size    # size of given dimension (in type units)
    uint32 stride  # stride of given dimension
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PolicyProbRequest(null);
    if (msg.obs !== undefined) {
      resolved.obs = new Array(msg.obs.length);
      for (let i = 0; i < resolved.obs.length; ++i) {
        resolved.obs[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.obs[i]);
      }
    }
    else {
      resolved.obs = []
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

class PolicyProbResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.mu = null;
      this.sigma = null;
    }
    else {
      if (initObj.hasOwnProperty('mu')) {
        this.mu = initObj.mu
      }
      else {
        this.mu = [];
      }
      if (initObj.hasOwnProperty('sigma')) {
        this.sigma = initObj.sigma
      }
      else {
        this.sigma = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type PolicyProbResponse
    // Serialize message field [mu]
    // Serialize the length for message field [mu]
    bufferOffset = _serializer.uint32(obj.mu.length, buffer, bufferOffset);
    obj.mu.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [sigma]
    // Serialize the length for message field [sigma]
    bufferOffset = _serializer.uint32(obj.sigma.length, buffer, bufferOffset);
    obj.sigma.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type PolicyProbResponse
    let len;
    let data = new PolicyProbResponse(null);
    // Deserialize message field [mu]
    // Deserialize array length for message field [mu]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.mu = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.mu[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [sigma]
    // Deserialize array length for message field [sigma]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.sigma = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.sigma[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.mu.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    object.sigma.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    return length + 8;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/PolicyProbResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5a3ab6a3c23a8c2cf2764eaadc06c44c';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    std_msgs/Float32MultiArray[] mu
    std_msgs/Float32MultiArray[] sigma
    
    
    ================================================================================
    MSG: std_msgs/Float32MultiArray
    # Please look at the MultiArrayLayout message definition for
    # documentation on all multiarrays.
    
    MultiArrayLayout  layout        # specification of data layout
    float32[]         data          # array of data
    
    
    ================================================================================
    MSG: std_msgs/MultiArrayLayout
    # The multiarray declares a generic multi-dimensional array of a
    # particular data type.  Dimensions are ordered from outer most
    # to inner most.
    
    MultiArrayDimension[] dim # Array of dimension properties
    uint32 data_offset        # padding elements at front of data
    
    # Accessors should ALWAYS be written in terms of dimension stride
    # and specified outer-most dimension first.
    # 
    # multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]
    #
    # A standard, 3-channel 640x480 image with interleaved color channels
    # would be specified as:
    #
    # dim[0].label  = "height"
    # dim[0].size   = 480
    # dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)
    # dim[1].label  = "width"
    # dim[1].size   = 640
    # dim[1].stride = 3*640 = 1920
    # dim[2].label  = "channel"
    # dim[2].size   = 3
    # dim[2].stride = 3
    #
    # multiarray(i,j,k) refers to the ith row, jth column, and kth channel.
    
    ================================================================================
    MSG: std_msgs/MultiArrayDimension
    string label   # label of given dimension
    uint32 size    # size of given dimension (in type units)
    uint32 stride  # stride of given dimension
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new PolicyProbResponse(null);
    if (msg.mu !== undefined) {
      resolved.mu = new Array(msg.mu.length);
      for (let i = 0; i < resolved.mu.length; ++i) {
        resolved.mu[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.mu[i]);
      }
    }
    else {
      resolved.mu = []
    }

    if (msg.sigma !== undefined) {
      resolved.sigma = new Array(msg.sigma.length);
      for (let i = 0; i < resolved.sigma.length; ++i) {
        resolved.sigma[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.sigma[i]);
      }
    }
    else {
      resolved.sigma = []
    }

    return resolved;
    }
};

module.exports = {
  Request: PolicyProbRequest,
  Response: PolicyProbResponse,
  md5sum() { return '543016ad28d3afef79460f66829d896a'; },
  datatype() { return 'tamp_ros/PolicyProb'; }
};
