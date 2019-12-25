import time
from std_msgs.msg import String, Bool
import rospy
start=time.time()
minutesgone=1
a=False
msg=Bool()
msg.data=True
pub = rospy.Publisher('still_alive', Bool, queue_size=10)
