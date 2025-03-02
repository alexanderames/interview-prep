def two_sum(nums, target)
  seen = {}  
  nums.each_with_index do |num, index|
    complement = target - num
    wtf = [seen[complement], index]
    puts "wtf: #{wtf}"
    return wtf if seen.key?(complement)
    seen[num] = index
  end
  []
end
puts two_sum([1,2,3,4,5,6,7,8,9], 4)