
lst = [1,3,2,5,3,6,2,7]

def quicksort(nums,l,r):
    if l>=r:
        return
    idx = partition(nums,l,r)
    quicksort(nums,l,idx-1)
    quicksort(nums,idx+1,r)

def partition(nums,l,r):
    pivot = nums[r]
    j = l
    for i in range(l,r):
        if nums[i]>pivot:
            nums[j],nums[i] = nums[i],nums[j]
            j += 1
    nums[r],nums[j] = nums[j],nums[r]
    return j

quicksort(lst,0,len(lst)-1)
print(lst)