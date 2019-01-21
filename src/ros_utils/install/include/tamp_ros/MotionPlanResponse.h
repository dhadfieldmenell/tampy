// Generated by gencpp from file tamp_ros/MotionPlanResponse.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_MOTIONPLANRESPONSE_H
#define TAMP_ROS_MESSAGE_MOTIONPLANRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Float32MultiArray.h>

namespace tamp_ros
{
template <class ContainerAllocator>
struct MotionPlanResponse_
{
  typedef MotionPlanResponse_<ContainerAllocator> Type;

  MotionPlanResponse_()
    : traj()
    , failed()
    , succes(false)  {
    }
  MotionPlanResponse_(const ContainerAllocator& _alloc)
    : traj(_alloc)
    , failed(_alloc)
    , succes(false)  {
  (void)_alloc;
    }



   typedef std::vector< ::std_msgs::Float32MultiArray_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::other >  _traj_type;
  _traj_type traj;

   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _failed_type;
  _failed_type failed;

   typedef uint8_t _succes_type;
  _succes_type succes;





  typedef boost::shared_ptr< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> const> ConstPtr;

}; // struct MotionPlanResponse_

typedef ::tamp_ros::MotionPlanResponse_<std::allocator<void> > MotionPlanResponse;

typedef boost::shared_ptr< ::tamp_ros::MotionPlanResponse > MotionPlanResponsePtr;
typedef boost::shared_ptr< ::tamp_ros::MotionPlanResponse const> MotionPlanResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tamp_ros::MotionPlanResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace tamp_ros

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'tamp_ros': ['/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "f62733bce50adbe17ca06f7359c6e15b";
  }

  static const char* value(const ::tamp_ros::MotionPlanResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xf62733bce50adbe1ULL;
  static const uint64_t static_value2 = 0x7ca06f7359c6e15bULL;
};

template<class ContainerAllocator>
struct DataType< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tamp_ros/MotionPlanResponse";
  }

  static const char* value(const ::tamp_ros::MotionPlanResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
std_msgs/Float32MultiArray[] traj\n\
string failed\n\
bool succes\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/Float32MultiArray\n\
# Please look at the MultiArrayLayout message definition for\n\
# documentation on all multiarrays.\n\
\n\
MultiArrayLayout  layout        # specification of data layout\n\
float32[]         data          # array of data\n\
\n\
\n\
================================================================================\n\
MSG: std_msgs/MultiArrayLayout\n\
# The multiarray declares a generic multi-dimensional array of a\n\
# particular data type.  Dimensions are ordered from outer most\n\
# to inner most.\n\
\n\
MultiArrayDimension[] dim # Array of dimension properties\n\
uint32 data_offset        # padding elements at front of data\n\
\n\
# Accessors should ALWAYS be written in terms of dimension stride\n\
# and specified outer-most dimension first.\n\
# \n\
# multiarray(i,j,k) = data[data_offset + dim_stride[1]*i + dim_stride[2]*j + k]\n\
#\n\
# A standard, 3-channel 640x480 image with interleaved color channels\n\
# would be specified as:\n\
#\n\
# dim[0].label  = \"height\"\n\
# dim[0].size   = 480\n\
# dim[0].stride = 3*640*480 = 921600  (note dim[0] stride is just size of image)\n\
# dim[1].label  = \"width\"\n\
# dim[1].size   = 640\n\
# dim[1].stride = 3*640 = 1920\n\
# dim[2].label  = \"channel\"\n\
# dim[2].size   = 3\n\
# dim[2].stride = 3\n\
#\n\
# multiarray(i,j,k) refers to the ith row, jth column, and kth channel.\n\
\n\
================================================================================\n\
MSG: std_msgs/MultiArrayDimension\n\
string label   # label of given dimension\n\
uint32 size    # size of given dimension (in type units)\n\
uint32 stride  # stride of given dimension\n\
";
  }

  static const char* value(const ::tamp_ros::MotionPlanResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.traj);
      stream.next(m.failed);
      stream.next(m.succes);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct MotionPlanResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tamp_ros::MotionPlanResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tamp_ros::MotionPlanResponse_<ContainerAllocator>& v)
  {
    s << indent << "traj[]" << std::endl;
    for (size_t i = 0; i < v.traj.size(); ++i)
    {
      s << indent << "  traj[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::stream(s, indent + "    ", v.traj[i]);
    }
    s << indent << "failed: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.failed);
    s << indent << "succes: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.succes);
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAMP_ROS_MESSAGE_MOTIONPLANRESPONSE_H
