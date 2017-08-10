import os
setup_file = ""
asm_file = ""
mia_file = ""

for file in os.listdir():
	if file.split(".")[1] == "asm":
		if asm_file != "":
			raise Exception("Cannot have duplicate assmeblerfiles")
		asm_file = file
	elif file.split(".")[1] == "set":
		if setup_file != "":
			raise Exception("Cannot have duplicate setupfiles")
		setup_file = file
	elif file.split(".")[1] == "mia":
		if mia_file != "":
			raise Exception("Cannot have duplicate miafiles")
		mia_file = file
if setup_file == "":
	raise Exception("No setup file")
if asm_file == "":
	raise Exception("No assembler file")
if mia_file == "":
	raise Exception("No mia file")
print("Files loaded:")
print("--------------------------------")
print(setup_file)
print(asm_file)
print(mia_file)
print("--------------------------------")

# Used to store instructions
class Instruction:
	def __init__(self, line):
		temp = line.split("\t")
		self.name = temp[0]
		self.addressing_modes = []
		for element in temp[1].split(","):
			self.addressing_modes.append(int(element))
		self.k1 = int(temp[2])
	def print_self(self):
		print("--------------")
		print("name: " + self.name)
		print("addressing modes: " + str(self.addressing_modes))
		print("k1 value: " + str(self.k1))
		print("--------------")

# loads in all the instructions from the setup file
op_codes = {}
setup_file_descriptor = open(setup_file)
for line in setup_file_descriptor.readlines():
	temp_op_code = Instruction(line)
	op_codes[temp_op_code.name] = temp_op_code
setup_file_descriptor.close()

# loads in the assembler file
asm_file_descriptor = open(asm_file)
instructions = []
for line in asm_file_descriptor.readlines():
	line_elements = line.split("\t")
	real_elements = []
	for element in line_elements:
		if element and element[0] != ";":
			if "\n" in element:
				real_elements.append(element.split("\n")[0])
			else:
				real_elements.append(element)
	if real_elements and real_elements[0]:
		instructions.append(real_elements)
pm = {}
counter = 0
names = {}
registers = {"G0":0, "G1":1, "G2":2, "G3":3}

for element in instructions:
	hex_counter = hex(counter).split("0x")[1]
	if len(hex_counter) == 1:
		hex_counter = "0" + hex_counter
	if element[0] == "SET":
		pm[element[1].lower()] = element[2]
		continue
	if ":" in element[0]:
		names[element[0].strip(":").lower()] = hex_counter
		continue
	if element[0] in ["LOAD", "ADD", "SUB", "AND", "CMP"]:
		op = op_codes[element[0]].k1
		reg = registers[element[1]]
		if "@" in element[2]:
			m = 2
		else:
			m = 0
		op = op << 4
		reg = reg << 2
		tot = op + reg + m
		tot = hex(tot).split("0x")[1]
		if len(tot) == 1:
			tot = "0" + tot
		tot += element[2].strip("@")
		pm[hex_counter] = tot
		counter += 1
		continue
	if element[0]=="STORE":
		op = op_codes[element[0]].k1
		reg = registers[element[1]]
		if "@" in element[2]:
			m = 2
		else:
			m = 0
		op = op << 4
		reg = reg << 2
		tot = op + reg + m
		tot = hex(tot).split("0x")[1]
		if len(tot) == 1:
			tot = "0" + tot
		tot += element[2].strip("@")
		pm[hex_counter] = tot
		counter += 1
		continue
	if element[0]=="LSR":
		op = op_codes[element[0]].k1
		reg = registers[element[1]]
		if "@" in element[2]:
			m = 2
		else:
			m = 0
		op = op << 4
		reg = reg << 2
		tot = op + reg + m
		tot = hex(tot).split("0x")[1]
		if len(tot) == 1:
			tot = "0" + tot
		tot += element[2].strip("@")
		pm[hex_counter] = tot
		counter += 1
		continue	
	if element[0] in ["BRA","BNE","BGE","BEQ"]:
		op = op_codes[element[0]].k1
		reg = 0
		m = 0
		op = op << 4
		tot = op + reg + m
		tot = hex(tot).split("0x")[1]
		if len(tot) == 1:
			tot = "0" + tot
		tot += "00"
		pm[hex_counter] = tot
		counter+=1
		hex_counter = hex(counter).split("0x")[1]
		if len(hex_counter) == 1:
			hex_counter = "0" + hex_counter
		pm[hex_counter] = element[1].lower()
		counter+=1
		continue
	if element[0]=="HALT":
		pm[hex_counter] = "8000"
		counter+=1
		continue
	else:
		raise Exception("something is fucky")		
		
	
	


# Write to all memory locations
output_file_descriptor = open("output.mia", "w")
output_file_descriptor.write("PM:\n")
for i in range(256):
	temp_hex_value = hex(i).split("0x")[1]
	if len(temp_hex_value) == 1:
		temp_hex_value = "0" + temp_hex_value
	if temp_hex_value not in pm:
		output_file_descriptor.write(temp_hex_value + ": 0000\n")
	else:
		if pm[temp_hex_value] not in names:
			output_file_descriptor.write(temp_hex_value + ": " + pm[temp_hex_value].lower() + "\n")
		else:
			output_file_descriptor.write(temp_hex_value + ": " + "00"+names[pm[temp_hex_value]] + "\n")

output_file_descriptor.write("\n")

# write the microcode to output
mia_file_descriptor = open(mia_file)
in_microcode_part = False
for line in mia_file_descriptor.readlines():
	if "MyM:" in line:
		in_microcode_part = True
	if in_microcode_part:
		if "PC:" in line:
			break
		else:
			output_file_descriptor.write(line)
# write state variables
output_file_descriptor.write("PC:\n00\n\nASR:\n00\n\nAR:\n0000\n\nHR:\n0000\n\nGR0:\n0000\n\nGR1:\n0000\n\nGR2:\n0000\n\nGR3:\n0000\n\nIR:\n0000\n\nMyPC:\n00\n\nSMyPC:\n00\n\nLC:\n00\n\nO_flag:\n\nC_flag:\n\nN_flag:\n\nZ_flag:\n\nL_flag:\nEnd_of_dump_file")
mia_file_descriptor.close()
# end of program
print("------------------------------------")
print("output.mia hes been generated")
print("------------------------------------")
output_file_descriptor.close()
		
	

			
	

