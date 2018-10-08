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

class MotionPlanResult {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.traj = null;
      this.failed = null;
      this.success = null;
      this.plan_id = null;
      this.cond = null;
      this.task = null;
      this.obj = null;
      this.targ = null;
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
      if (initObj.hasOwnProperty('success')) {
        this.success = initObj.success
      }
      else {
        this.success = false;
      }
      if (initObj.hasOwnProperty('plan_id')) {
        this.plan_id = initObj.plan_id
      }
      else {
        this.plan_id = 0;
      }
      if (initObj.hasOwnProperty('cond')) {
        this.cond = initObj.cond
      }
      else {
        this.cond = 0;
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
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type MotionPlanResult
    // Serialize message field [traj]
    // Serialize the length for message field [traj]
    bufferOffset = _serializer.uint32(obj.traj.length, buffer, bufferOffset);
    obj.traj.forEach((val) => {
      bufferOffset = std_msgs.msg.Float32MultiArray.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [failed]
    bufferOffset = _serializer.string(obj.failed, buffer, bufferOffset);
    // Serialize message field [success]
    bufferOffset = _serializer.bool(obj.success, buffer, bufferOffset);
    // Serialize message field [plan_id]
    bufferOffset = _serializer.int32(obj.plan_id, buffer, bufferOffset);
    // Serialize message field [cond]
    bufferOffset = _serializer.int32(obj.cond, buffer, bufferOffset);
    // Serialize message field [task]
    bufferOffset = _serializer.string(obj.task, buffer, bufferOffset);
    // Serialize message field [obj]
    bufferOffset = _serializer.string(obj.obj, buffer, bufferOffset);
    // Serialize message field [targ]
    bufferOffset = _serializer.string(obj.targ, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type MotionPlanResult
    let len;
    let data = new MotionPlanResult(null);
    // Deserialize message field [traj]
    // Deserialize array length for message field [traj]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.traj = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.traj[i] = std_msgs.msg.Float32MultiArray.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [failed]
    data.failed = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [success]
    data.success = _deserializer.bool(buffer, bufferOffset);
    // Deserialize message field [plan_id]
    data.plan_id = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [cond]
    data.cond = _deserializer.int32(buffer, bufferOffset);
    // Deserialize message field [task]
    data.task = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [obj]
    data.obj = _deserializer.string(buffer, bufferOffset);
    // Deserialize message field [targ]
    data.targ = _deserializer.string(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    object.traj.forEach((val) => {
      length += std_msgs.msg.Float32MultiArray.getMessageSize(val);
    });
    length += object.failed.length;
    length += object.task.length;
    length += object.obj.length;
    length += object.targ.length;
    return length + 29;
  }

  static datatype() {
    // Returns string type for a message object
    return 'tamp_ros/MotionPlanResult';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '326cd4f386f413012deeafd4217bb17d';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
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
    const resolved = new MotionPlanResult(null);
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

    if (msg.success !== undefined) {
      resolved.success = msg.success;
    }
    else {
      resolved.success = false
    }

    if (msg.plan_id !== undefined) {
      resolved.plan_id = msg.plan_id;
    }
    else {
      resolved.plan_id = 0
    }

    if (msg.cond !== undefined) {
      resolved.cond = msg.cond;
    }
    else {
      resolved.cond = 0
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

    return resolved;
    }
};

module.exports = MotionPlanResult;
