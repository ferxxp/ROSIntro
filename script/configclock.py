import rospy
from interaccion.msg import usuario
from std_msgs.msg import String
sent_first_message=False
msgclock=String()
msgclock.data="0x11"
pubstart = rospy.Publisher('start_topic', String, queue_size=10)
pubreset = rospy.Publisher('reset_topic', String, queue_size=10)
