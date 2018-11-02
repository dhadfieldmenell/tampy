// Auto-generated. Do not edit!

// (in-package tamp_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let MotionPlanResult = require('./MotionPlanResult.js');

//-----------------------------------------------------------

class HLPlanResult {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.steps = null;
      this.path_to = null;
      this.success = null;
      this.cond = null;
    }
    else {
      if (initObj.hasOwnProperty('steps')) {
        this.steps = initObj.steps
      }
      else {
        this.steps = [];
      }
      if (initObj.hasOwnProperty('path_to')) {
        this.path_to = initObj.path_to
      }
      else {
        this.path_to = '';
      }
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('cond')) {
        this.cond = initObj.cond
      }
      else {
        this.cond = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type HLPlanResult
    // Serialize message field [steps]
    // Serialize the length for message field [steps]
    bufferOffset = _serializer.uint32(obj.steps.length, buffer, bufferOffset);
    obj.steps.forEach((val) => {
      bufferOffset = MotionPlanResult.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [path_to]
    bufferOffset = _serializer.string(obj.path_to, buffer, bufferOffset);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [cond]
    bufferOffset = _serializer.int32(obj.cond, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type HLPlanResult
    let len;
    let data = new HLPlanResult(null);
    // Deserialize message field [steps]
    // Deserialize array length for message field [steps]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.steps = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.steps[i] = MotionPlanResult.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [path_to]
    data.path_to = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [cond]
    data.cond = _deserializer.int32(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.steps.forEach((val) => {
      length += MotionPlanResult.getMessageSize(val);
    });
    length += object.path_to.length;
    return length + 13;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/HLPlanResult';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '19bff39c2204ab093accc09544e93f76';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    MotionPlanResult[] steps
    string path_to
    bool success
    int32 cond
    
    ================================================================================
    MSG: tamp_ros/MotionPlanResult
    std_msgs/Float32MultiArray[] traj
    string failed
    bool success
    int32 plan_id
    int32 cond
    string task
    string obj
    string targ
    
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
    const resolved = new HLPlanResult(null);
    if (msg.steps !== undefined) {
      resolved.steps = new Array(msg.steps.length);
      for (let i = 0; i < resolved.steps.length; ++i) {
        resolved.steps[i] = MotionPlanResult.Resolve(msg.steps[i]);
      }
    }
    else {
      resolved.steps = []
    }

    if (msg.path_to !== undefined) {
      resolved.path_to = msg.path_to;
    }
    else {
      resolved.path_to = ''
    }

    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.cond !== undefined) {
      resolved.cond = msg.cond;
    }
    else {
      resolved.cond = 0
    }

    return resolved;
    }
};

module.exports = HLPlanResult;
