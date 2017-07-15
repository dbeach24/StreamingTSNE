
using SFML

const window_width = 1920 * 2
const window_height = 1080 * 2

function mkdot(x, y; r=10, color=SFML.red)
    dot = CircleShape()
    set_position(dot, Vector2f(x, y))
    set_radius(dot, r)
    set_fillcolor(dot, color)
    set_origin(dot, Vector2f(r, r))
    dot
end


function create_dots(N)
    [mkdot(rand(0:window_width), rand(0:window_height), r=3) for i in 1:N]
end

function move_dots(dots)
    for dot in dots
        x, y = get_position(dot)
        x = x + rand(-2:2)
        y = y + rand(-2:2)
        set_position(dot, Vector2f(x, y))
    end
end

function main()

    window = RenderWindow("Dots", window_width, window_height)
    set_framerate_limit(window, 60)
    set_vsync_enabled(window, true)

    event = Event()

    dots = create_dots(20000)

    while isopen(window)
        while pollevent(window, event)
            if get_type(event) == EventType.CLOSED
                close(window)
            end
        end

        move_dots(dots)

        clear(window, SFML.white)
        for dot in dots
            draw(window, dot)
        end

        display(window)
    end

end

main()
