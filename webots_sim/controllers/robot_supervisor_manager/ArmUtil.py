class ToArmCoord:
    """
    Convert from world coordinate (x, y, z)
    to arm coordinate (x, -z, y)
    """
    @staticmethod
    def convert(worldCoord):
        """
        arg:
            worldCoord: [x, y, z]
                An array of 3 containing the 3 world coordinate.
        """
        return [worldCoord[0], -worldCoord[2], worldCoord[1]]

class Func:
    @staticmethod
    def getValue(positionSensorList):
        psValue = []
        for i in positionSensorList:
            psValue.append(i.getValue())
        return psValue


    @staticmethod
    def get_All_motors(robot):
        """
        Get 7 motors from the robot model
        """
        motorList = []
        # NOTE this is a quick fix for UR10e
        motorList.append(robot.getDevice('shoulder_pan_joint'))
        motorList.append(robot.getDevice('shoulder_lift_joint'))
        motorList.append(robot.getDevice('elbow_joint'))
        motorList.append(robot.getDevice('wrist_1_joint'))
        motorList.append(robot.getDevice('wrist_2_joint'))
        motorList.append(robot.getDevice('wrist_3_joint'))
#		for i in range(7):
#			motorName = 'motor' + str(i + 1)
#			motor = robot.getDevice(motorName)	 # Get the motor handle #positionSensor1
#			motor.setPosition(float('inf'))  # Set starting position
#			motor.setVelocity(0.0)  # Zero out starting velocity
#			motorList.append(motor)
        return motorList
		
    @staticmethod
    def get_All_positionSensors(robot, timestep):
        """
        Get 7 position sensors from the robot model
        """
        positionSensorList = []
        # NOTE this is a quick fix for UR10e

        positionSensorList.append(robot.getDevice('shoulder_pan_joint_sensor'))
        positionSensorList.append(robot.getDevice('shoulder_lift_joint_sensor'))
        positionSensorList.append(robot.getDevice('elbow_joint_sensor'))
        positionSensorList.append(robot.getDevice('wrist_1_joint_sensor'))
        positionSensorList.append(robot.getDevice('wrist_2_joint_sensor'))
        positionSensorList.append(robot.getDevice('wrist_3_joint_sensor'))

        for sensor in positionSensorList:
        #sensor.enable(10)
            sensor.enable(timestep)
#		for i in range(7):
#			positionSensorName = 'positionSensor' + str(i+1)
#			positionSensor = robot.getDevice(positionSensorName)
#			positionSensor.enable(timestep)
#			positionSensorList.append(positionSensor)
        return positionSensorList
