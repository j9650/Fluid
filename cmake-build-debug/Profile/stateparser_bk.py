import sys
import re
states = {}
total = {}
def main():
	f = open(sys.argv[1], 'r')
	for line in f:
		sp = line.split()
		if "===" in line:
			continue
		elif "enter" in sp:
			[task, _2, _3] = re.split('(\d+)',sp[0])
			[state, timestamp, _3] =   re.split('(\d+)',sp[-1])
			#print("enter", sp, task, state, timestamp)
			if task not in states: states[task] = {}
			states[task][state] = timestamp
		elif "left" in sp: 
			[task, _2, _3] = re.split('(\d+)',sp[0])
			[state, timestamp, _3] = re.split('(\d+)',sp[-1])
			#print("left", sp, task, state, timestamp)
			if task not in total: total[task] = {}
			if state not in total[task]: total[task][state] = []
			total[task][state].append(int(timestamp) - int(states[task][state]))
	for task in total:
		for state in total[task]:
			print (task, state, len(total[task][state]), min(total[task][state]), max(total[task][state]), sum(total[task][state])/len(total[task][state]))

main()