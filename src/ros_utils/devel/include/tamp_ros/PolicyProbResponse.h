// Generated by gencpp from file tamp_ros/PolicyProbResponse.msg
// DO NOT EDIT!


#ifndef TAMP_ROS_MESSAGE_POLICYPROBRESPONSE_H
#define TAMP_ROS_MESSAGE_POLICYPROBRESPONSE_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float32MultiArray.h>

namespace tamp_ros
{
template <class ContainerAllocator>
struct PolicyProbResponse_
{
  typedef PolicyProbResponse_<ContainerAllocator> Type;

  PolicyProbResponse_()
    : mu()
    , sigma()  {
    }
  PolicyProbResponse_(const ContainerAllocator& _alloc)
    : mu(_alloc)
    , sigma(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector< ::std_msgs::Float32MultiArray_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::other >  _mu_type;
  _mu_type mu;

   typedef std::vector< ::std_msgs::Float32MultiArray_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::other >  _sigma_type;
  _sigma_type sigma;





  typedef boost::shared_ptr< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> const> ConstPtr;

}; // struct PolicyProbResponse_

typedef ::tamp_ros::PolicyProbResponse_<std::allocator<void> > PolicyProbResponse;

typedef boost::shared_ptr< ::tamp_ros::PolicyProbResponse > PolicyProbResponsePtr;
typedef boost::shared_ptr< ::tamp_ros::PolicyProbResponse const> PolicyProbResponseConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::tamp_ros::PolicyProbResponse_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace tamp_ros

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': False, 'IsMessage': True, 'HasHeader': False}
// {'tamp_ros': ['/home/michaelmcdonald/dependencies/tampy/src/ros_utils/src/tamp_ros/msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5a3ab6a3c23a8c2cf2764eaadc06c44c";
  }

  static const char* value(const ::tamp_ros::PolicyProbResponse_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5a3ab6a3c23a8c2cULL;
  static const uint64_t static_value2 = 0xf2764eaadc06c44cULL;
};

template<class ContainerAllocator>
struct DataType< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "tamp_ros/PolicyProbResponse";
  }

  static const char* value(const ::tamp_ros::PolicyProbResponse_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
{
  static const char* value()
  {
    return "\n\
std_msgs/Float32MultiArray[] mu\n\
std_msgs/Float32MultiArray[] sigma\n\
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

  static const char* value(const ::tamp_ros::PolicyProbResponse_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.mu);
      stream.next(m.sigma);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct PolicyProbResponse_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::tamp_ros::PolicyProbResponse_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::tamp_ros::PolicyProbResponse_<ContainerAllocator>& v)
  {
    s << indent << "mu[]" << std::endl;
    for (size_t i = 0; i < v.mu.size(); ++i)
    {
      s << indent << "  mu[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::stream(s, indent + "    ", v.mu[i]);
    }
    s << indent << "sigma[]" << std::endl;
    for (size_t i = 0; i < v.sigma.size(); ++i)
    {
      s << indent << "  sigma[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::std_msgs::Float32MultiArray_<ContainerAllocator> >::stream(s, indent + "    ", v.sigma[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // TAMP_ROS_MESSAGE_POLICYPROBRESPONSE_H