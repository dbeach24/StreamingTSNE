
using SFML
using HDF5

const window_width = 1000
const window_height = 1000

function mkdot(x, y; r=2, color=SFML.red)
    dot = CircleShape()
    set_position(dot, Vector2f(x, y))
    set_radius(dot, r)
    set_fillcolor(dot, color)
    set_origin(dot, Vector2f(r, r))
    dot
end

function do_display(points)

    window = RenderWindow("Scatter", window_width, window_height)
    set_framerate_limit(window, 60)
    set_vsync_enabled(window, true)

    event = Event()

    while isopen(window)
        while pollevent(window, event)
            if get_type(event) == EventType.CLOSED
                close(window)
            end
        end

        clear(window, SFML.white)

        for point in points
            draw(window, point)
        end

        display(window)
    end

end

function main()

    infile = ARGS[1]

    Y = map(Float64, h5read(infile, "Y")')
    labels = h5read(infile, "labels")

    points = Vector{CircleShape}()
    N, k = size(Y)
    minx = minimum(Y[:,1])
    maxx = maximum(Y[:,1])
    miny = minimum(Y[:,2])
    maxy = maximum(Y[:,2])

    xmult = window_width / (maxx - minx)
    ymult = window_height / (maxy - miny)

    xshift = (window_width - (maxx - minx)) / 2
    yshift = (window_height - (maxy - miny)) / 2

    colors = [
        Color(166,206,227),
        Color(31,120,180),
        Color(178,223,138),
        Color(51,160,44),
        Color(251,154,153),
        Color(227,26,28),
        Color(253,191,111),
        Color(255,127,0),
        Color(202,178,214),
        Color(106,61,154),
    ]

    @assert k == 2
    for i=1:N
        x = Y[i,1] * xmult + xshift
        y = Y[i,2] * ymult + yshift
        c = Int32(labels[i]) + 1
        point = mkdot(x, y; color=colors[c])
        push!(points, point)
    end

    do_display(points)

end

main()


