import sys
import re
def main():
	f = open(sys.argv[1], 'r')
	id = 0
	task_record = []
	state_list = []
	state_access = []
	"""
	for line in f:
		sp = line.strip().split()
		if "===" in line:
			id = id + 1
			continue
		elif "enter" in sp:
			ss = sp.split("::")
			task_name = ss[0][:-4]
			if task_name not in task_names:
				task_names.append(task_name)
				task_lists[task_name] = []
				total[task_name] = {}
			task_lists[task_name].append([])
			state = sp[2]
			if state not in state_dic:
				state_dic.append(state)
			if state not in total[task_name]:
				total[task_name][state] = []
				for i in range(6)
				total[task_name][state].append(0.0)
			timestamp = "-"+sp[3]
			#print("enter", sp, task, state, timestamp)
			one = []
			one.append(state)
			one.append(timestamp)
			task_lists[task_name][id].append(one)
		elif "left" in sp: 
			ss = sp.split("::")
			task_name = ss[0][:-4]
			state = sp[2]
			one = task_lists[task_name][id][-1]
			timestamp = int(sp[3])
			one[1] = str(timestamp + int(one[1]))
			#print("enter", sp, task, state, timestamp)
			task_lists[task_name][id][-1] = one
	"""
	total = {}
	task_names = []
	task_record.append([])
	for line in f:
		sp = line.strip().split()
		if "===" in line:
			id = id + 1
			task_record.append([])
			continue
		elif "enter" in sp:
			ss = sp[0].split("::")
			task_name = ss[0][:6]
			[task_name, _2, _3] = re.split('(\d+)',sp[0])
			if task_name not in task_names:
				task_names.append(task_name)
				total[task_name] = {}

			state = sp[2]

			if state not in total[task_name]:
				total[task_name][state] = []
				for i in range(8):
					total[task_name][state].append(0.0)
				total[task_name][state][0] = 100000000.0
				total[task_name][state][4] = 100000000.0

			if state+" "+task_name not in state_list:
				state_list.append(state+" "+task_name)
				state_access.append(0)
			timestamp = "-"+sp[3]
			one = []
			one.append(task_name)
			one.append(state)
			one.append(timestamp)
			task_record[id].append(one)
		elif "left" in sp:
			one = task_record[id][-1]
			ss = sp[0].split("::")
			task_name = ss[0][:-4]
			[task_name, _2, _3] = re.split('(\d+)',sp[0])
			state = sp[2]
			timestamp = int(sp[3])
			one[2] = str(timestamp + int(one[2]))
			task_record[id][-1] = one

	for i in range(len(task_record)):
		record = task_record[i]

		for t in range(len(state_list)):
			state_access[t] = 0;
		for one in record:
			task_name = one[0]
			state = one[1]
			timestamp = int(one[2])
			if(timestamp < 0) :
				continue
			index = state_list.index(state+" "+task_name)
			state_access[index] = state_access[index] + 1
			if timestamp > total[task_name][state][5]:
				total[task_name][state][5] = timestamp
			if timestamp < total[task_name][state][4]:
				total[task_name][state][4] = timestamp
			total[task_name][state][6] = total[task_name][state][6] + timestamp
			total[task_name][state][7] = total[task_name][state][7] + 1
			# print(state, ' ', task_name, ' ', str(timestamp))

		for t in range(len(state_list)):
			state = state_list[t].split()[0]
			task_name = state_list[t].split()[1]
			access_time = state_access[t]
			if (access_time == 0):
				continue
			if access_time > total[task_name][state][1]:
				total[task_name][state][1] = access_time
			if access_time < total[task_name][state][0]:
				total[task_name][state][0] = access_time
			total[task_name][state][2] = total[task_name][state][2] + access_time
			total[task_name][state][3] = total[task_name][state][3] + 1
			# print(state, ' ', task_name, ' ', str(access_time))


	for task in total:
		for state in total[task]:
			print (task, state, total[task][state][0], total[task][state][1], total[task][state][2]/total[task][state][3], total[task][state][4], total[task][state][5], total[task][state][6]/total[task][state][7])

main()