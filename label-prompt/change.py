input_file = 'cifar100_labels.txt'  # 수정할 파일 경로
output_file = 'cifar100_labels_a.txt'  # 결과를 저장할 파일 경로

# 파일 읽기
with open(input_file, 'r') as file:
    lines = file.readlines()

# 각 줄 끝에 문장을 추가
with open(output_file, 'w') as file:
    for line in lines:
        file.write(line.strip() + " from cifar100\n")

print(f"수정된 파일이 {output_file}에 저장되었습니다.")