import re

# 입력 파일과 출력 파일의 이름을 지정합니다.
input_file = "sorted_imagenet_classes.txt"
output_file = "imagenet-1k.txt"

# 데이터를 저장할 리스트를 초기화합니다.
parsed_data = []

def process_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(' ', 2)
            if len(parts) == 3:
                f_out.write(parts[2] + '\n')


process_file(input_file, output_file)
print(f"처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")