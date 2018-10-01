// Auto-generated. Do not edit!

// (in-package tamp_ros.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let std_msgs = _finder('std_msgs');

//-----------------------------------------------------------

class MotionPlanProblem {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.prob_id = null;
      this.task = null;
      this.obj = null;
      this.targ = null;
      this.state = null;
      this.cond = null;
      this.traj_mean = null;
    }
    else {
      if (initObj.hasOwnProperty('prob_id')) {
        this.prob_id = initObj.prob_id
      }
      else {
        this.prob_id = 0;
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
      if (initObj.hasOwnProperty('state')) {
        this.state = initObj.state
      }
      else {
        this.state = [];
      }
      if (initObj.hasOwnProperty('cond')) {
        this.cond = initObj.cond
      }
      else {
        this.cond = 0;
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
    // Serializes a message object of type MotionPlanProblem
    // Serialize message field [prob_id]
    bufferOffset = _serializer.int32(obj.prob_id, buffer, bufferOffset);
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [obj]
    bufferOffset = _serializer.string(obj.obj, buffer, bufferOffset);
    // Serialize message field [targ]
    bufferOffset = _serializer.string(obj.targ, buffer, bufferOffset);
    // Serialize message field [state]
    bufferOffset = _arraySerializer.float32(obj.state, buffer, bufferOffset, null);
    // Serialize message field [cond]
    bufferOffset = _serializer.int32(obj.cond, buffer, bufferOffset);
    // Serialize message field [traj_mean]
    // Serialize the length for message field [traj_mean]
    bufferOffset = _serializer.uint32(obj.traj_mean.length, buffer, bufferOffset);
    obj.traj_mean.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MotionPlanProblem
    let len;
    let data = new MotionPlanProblem(null);
    // Deserialize message field [prob_id]
    data.prob_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [obj]
    data.obj = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [targ]
    data.targ = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [state]
    data.state = _arrayDeserializer.float32(buffer, bufferOffset, null)
    // Deserialize message field [cond]
    data.cond = _deserializer.int32(buffer, bufferOffset);
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
    length += object.task.length;
    length += object.obj.length;
    length += object.targ.length;
    length += 4 * object.state.length;
    object.traj_mean.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    return length + 28;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/MotionPlanProblem';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'fd42f7570f643f3f3f8d1c5e9bc3c462';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    int32 prob_id
    string task
    string obj
    string targ
    float32[] state
    int32 cond
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
    const resolved = new MotionPlanProblem(null);
    if (msg.prob_id !== undefined) {
      resolved.prob_id = msg.prob_id;
    }
    else {
      resolved.prob_id = 0
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

    if (msg.state !== undefined) {
      resolved.state = msg.state;
    }
    else {
      resolved.state = []
    }

    if (msg.cond !== undefined) {
      resolved.cond = msg.cond;
    }
    else {
      resolved.cond = 0
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

module.exports = MotionPlanProblem;
