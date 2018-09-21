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

class MotionPlanRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.state = null;
      this.task = null;
      this.obj = null;
      this.targ = null;
      this.condition = null;
      this.traj_mean = null;
    }
    else {
      if (initObj.hasOwnProperty('state')) {
        this.state = initObj.state
      }
      else {
        this.state = [];
      }
      if (initObj.hasOwnProperty('task')) {
        this.task = initObj.task
      }
      else {
        this.task = '';
      }
      if (initObj.hasOwnProperty('obj')) {
        this.obj = initObj.obj
      }
      else {
        this.obj = '';
      }
      if (initObj.hasOwnProperty('targ')) {
        this.targ = initObj.targ
      }
      else {
        this.targ = '';
      }
      if (initObj.hasOwnProperty('condition')) {
        this.condition = initObj.condition
      }
      else {
        this.condition = 0;
      }
      if (initObj.hasOwnProperty('traj_mean')) {
        this.traj_mean = initObj.traj_mean
      }
      else {
        this.traj_mean = [];
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MotionPlanRequest
    // Serialize message field [state]
    bufferOffset = _arraySerializer.float32(obj.state, buffer, bufferOffset, null);
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [obj]
    bufferOffset = _serializer.string(obj.obj, buffer, bufferOffset);
    // Serialize message field [targ]
    bufferOffset = _serializer.string(obj.targ, buffer, bufferOffset);
    // Serialize message field [condition]
    bufferOffset = _serializer.int32(obj.condition, buffer, bufferOffset);
    // Serialize message field [traj_mean]
    // Serialize the length for message field [traj_mean]
    bufferOffset = _serializer.uint32(obj.traj_mean.length, buffer, bufferOffset);
    obj.traj_mean.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MotionPlanRequest
    let len;
    let data = new MotionPlanRequest(null);
    // Deserialize message field [state]
    data.state = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [obj]
    data.obj = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [targ]
    data.targ = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [condition]
    data.condition = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [traj_mean]
    // Deserialize array length for message field [traj_mean]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.traj_mean = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.traj_mean[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 4 * object.state.length;
    length += object.task.length;
    length += object.obj.length;
    length += object.targ.length;
    object.traj_mean.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    return length + 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/MotionPlanRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '5694d27d1fc584cff166faf3dd6d7011';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    float32[] state
    string task
    string obj
    string targ
    int32 condition
    std_msgs/Float32MultiArray[] traj_mean
    
    
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
    const resolved = new MotionPlanRequest(null);
    if (msg.state !== undefined) {
      resolved.state = msg.state;
    }
    else {
      resolved.state = []
    }

    if (msg.task !== undefined) {
      resolved.task = msg.task;
    }
    else {
      resolved.task = ''
    }

    if (msg.obj !== undefined) {
      resolved.obj = msg.obj;
    }
    else {
      resolved.obj = ''
    }

    if (msg.targ !== undefined) {
      resolved.targ = msg.targ;
    }
    else {
      resolved.targ = ''
    }

    if (msg.condition !== undefined) {
      resolved.condition = msg.condition;
    }
    else {
      resolved.condition = 0
    }

    if (msg.traj_mean !== undefined) {
      resolved.traj_mean = new Array(msg.traj_mean.length);
      for (let i = 0; i < resolved.traj_mean.length; ++i) {
        resolved.traj_mean[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.traj_mean[i]);
      }
    }
    else {
      resolved.traj_mean = []
    }

    return resolved;
    }
};

class MotionPlanResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.traj = null;
      this.failed = null;
      this.succes = null;
    }
    else {
      if (initObj.hasOwnProperty('traj')) {
        this.traj = initObj.traj
      }
      else {
        this.traj = [];
      }
      if (initObj.hasOwnProperty('failed')) {
        this.failed = initObj.failed
      }
      else {
        this.failed = '';
      }
      if (initObj.hasOwnProperty('succes')) {
        this.succes = initObj.succes
      }
      else {
        this.succes = false;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MotionPlanResponse
    // Serialize message field [traj]
    // Serialize the length for message field [traj]
    bufferOffset = _serializer.uint32(obj.traj.length, buffer, bufferOffset);
    obj.traj.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [failed]
    bufferOffset = _serializer.string(obj.failed, buffer, bufferOffset);
    // Serialize message field [succes]
    bufferOffset = _serializer.bool(obj.succes, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MotionPlanResponse
    let len;
    let data = new MotionPlanResponse(null);
    // Deserialize message field [traj]
    // Deserialize array length for message field [traj]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.traj = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.traj[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [failed]
    data.failed = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [succes]
    data.succes = _deserializer.bool(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.traj.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    length += object.failed.length;
    return length + 9;
  }

  static datatype() {
    // Returns string type for a service object
    return 'tamp_ros/MotionPlanResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'f62733bce50adbe17ca06f7359c6e15b';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    
    std_msgs/Float32MultiArray[] traj
    string failed
    bool succes
    
    
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
    const resolved = new MotionPlanResponse(null);
    if (msg.traj !== undefined) {
      resolved.traj = new Array(msg.traj.length);
      for (let i = 0; i < resolved.traj.length; ++i) {
        resolved.traj[i] = std_msgs.msg.Float32MultiArray.Resolve(msg.traj[i]);
      }
    }
    else {
      resolved.traj = []
    }

    if (msg.failed !== undefined) {
      resolved.failed = msg.failed;
    }
    else {
      resolved.failed = ''
    }

    if (msg.succes !== undefined) {
      resolved.succes = msg.succes;
    }
    else {
      resolved.succes = false
    }

    return resolved;
    }
};

module.exports = {
  Request: MotionPlanRequest,
  Response: MotionPlanResponse,
  md5sum() { return '3979cf2783d6a8eedc4996bbfe7b87c9'; },
  datatype() { return 'tamp_ros/MotionPlan'; }
};
