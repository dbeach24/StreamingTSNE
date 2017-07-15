module StatLRU

type StatLRUCache{K, V}
    maxdepth::Int
    maxcapacity::Int
    pcount::Int
    d::Dict{K,Tuple{Int,V}}
end

function setitem!(cache::StatLRUCache{K, V}, K, V)

end



function demote!(cache::StatLRUCache)
    # no need to demote if we haven't hit capacity
    pcount < maxcapacity && return
    # pick a random key


    # decrement its level

    if level == 1
        

    # decrement pcount




end




end # module