import sys
import subprocess
print("Welcome to the inverse kinematics algoritms testing program!")
print("")
print("There is a version with and without joint limits.")
print("You can choose between the damped squares, jacobian traspose or jacobian inverse algorithms.")
print("The algorithms are optimized with regards to the (non-)existing constraints.")
print("The robot is specified by its Denavit-Hartenberg parameters in the robot_parameters file.")
print("To close this program, type CTRL+C")
while(1):
    while(1):
        const = input("Do you want the robot to have joint constaints (3/4 of a circle)? [Y/n]: ")
        print(const)
        if (const == 'Y') or (const == 'y'):
            const = '1'
            break
        else:
            if (const == 'N') or (const == 'n'):
                const = '0'
                break
            else:
                print("The input is not in the expected format. Try again.")

    while(1):
        target = input("Please enter the target end effector position in the x.x,y.y,z.z format: ")
        try:
            e = "ok"
            target_l = target.split(',')
            for i in range(len(target_l)):
                target_l[i] = float(target_l[i])
        except:
            e = sys.exc_info()
            print("The input is not in the expected format. Try again.")
            print(e)
        if e == "ok":
            break

    while(1):
        method = input("Please specify the desired algorithm:\n'T' for transpose \
                         \n'P' for pseudoinverse\n'S' for damped squares\n'G' for graddesc: ")
        if (method == 'T') or (method == 'P') or  (method == 'S') or (method =='G'):
            proc = subprocess.run(["python3", "inv_kinm.py", target, method, const])
            break
        else:
            print("The input is not in the expected format. Try again.")
        
