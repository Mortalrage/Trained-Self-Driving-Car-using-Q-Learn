import os
import logging
import pyglet

# Enable helpful logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# --- imports from your project (may raise ImportError if PYTHONPATH wrong) ---
try:
    from models.objects.track import Track
    from models.objects.car import Car
    from controllers.evaluators.line_evaluator import LineEvaluator
    from controllers.drivers.deep_q_driver import DeepQDriver
    from controllers.drivers.player_driver import PlayerDriver
    from views.drawer import Drawer
except Exception as e:
    log.exception("Failed importing project modules: %s", e)
    raise

# --- paths and existence checks ---
TRACK_PATH = "assets/tracks/track-1.trk"
EVAL_LINES_PATH = "assets/evaluator_lines/track-1.lns"
MODEL_PATH = "model.h5"  # adjust if your file is elsewhere

for p in (TRACK_PATH, EVAL_LINES_PATH):
    if not os.path.exists(p):
        log.error("Required asset missing: %s\nRun from project root or fix the path.", p)
        raise FileNotFoundError(p)

# Load track and initialize car
try:
    track = Track.load_from_file(TRACK_PATH)
except Exception as e:
    log.exception("Failed to load track file '%s': %s", TRACK_PATH, e)
    raise

car = Car(width=24, height=45)
try:
    car.init_position(track.start_point, track.start_direction)
except Exception as e:
    log.exception("Failed to initialize car position: %s", e)
    # Try a safe fallback if possible
    try:
        car.init_position(track.start_point, track.start_direction.normalized())
    except Exception:
        raise

# Create evaluator (wrap in try so we can diagnose format problems)
try:
    evaluator = LineEvaluator.load_lines_from_file(car, track, EVAL_LINES_PATH)
except Exception as e:
    log.exception("Failed to create LineEvaluator from '%s': %s", EVAL_LINES_PATH, e)
    # If load_lines_from_file is failing for file format reasons, raise so user can fix file
    raise

# Create driver and try loading model weights; if model not found, continue but warn
driver = DeepQDriver(accepted_sensors=10, layer_count=8, output_per_hidden=32)

if os.path.exists(MODEL_PATH):
    try:
        driver.load_model_weights(MODEL_PATH)
        log.info("Model weights loaded from %s", MODEL_PATH)
    except Exception as e:
        log.exception("Failed to load model weights from '%s': %s", MODEL_PATH, e)
        log.warning("Continuing without pretrained weights. If this is unintended, check the model file or loader.")
else:
    log.warning("Model file not found at '%s'. Running driver with randomly initialized policy.", MODEL_PATH)

# Drawer + pyglet window
display_w, display_h = 1280, 720
drawer = Drawer(display_w, display_h, car, track, evaluator)

# Create window and player driver (attach keyboard events)
window = pyglet.window.Window(display_w, display_h, resizable=True)
p_driver = PlayerDriver(window)  # assume its constructor binds event handlers to the window

# In case PlayerDriver requires explicit registration, attempt to bind
if hasattr(p_driver, "register") and callable(getattr(p_driver, "register")):
    try:
        p_driver.register(window)
    except Exception:
        # many PlayerDriver implementations register inside __init__, so it's optional
        pass

start_direction = track.start_direction
try:
    start_direction = start_direction.normalized()
except Exception:
    # if normalized not available/correct, keep original
    pass

# Game loop
def game_loop(dt: float):
    global track, car, p_driver, evaluator, start_direction

    try:
        # get state and action safely; driver may raise if model not loaded
        state = None
        try:
            state = driver.get_input_data(car, track)
        except Exception as e:
            log.debug("driver.get_input_data() failed (non-fatal): %s", e)

        try:
            turn_rate, acceleration_rate, action = driver.calculate_command(car, track, False)
        except Exception as e:
            # fallback behavior: no steering, no accel
            log.warning("driver.calculate_command() failed; using safe zero action. Error: %s", e)
            turn_rate, acceleration_rate, action = 0.0, 0.0, None

        # Move car
        try:
            car.move(turn_rate, acceleration_rate, dt)
        except Exception as e:
            log.exception("car.move() failed: %s", e)
            # re-init to safe start to continue running
            car.init_position(track.start_point, start_direction)
            return

        # Evaluate
        try:
            done = not evaluator.evaluate(dt)
        except Exception as e:
            log.exception("evaluator.evaluate() error: %s", e)
            done = True

        if done:
            log.info("Episode done → resetting car & evaluator")
            try:
                car.init_position(track.start_point, start_direction)
            except Exception as e:
                log.exception("Failed to re-init car: %s", e)
            try:
                evaluator.reset_score()
            except Exception as e:
                log.exception("Failed to reset evaluator score: %s", e)

    except Exception as e:
        # Catch-all for unexpected errors so pyglet loop doesn't crash silently
        log.exception("Unhandled error inside game loop: %s", e)
        # attempt to reset to safe state
        try:
            car.init_position(track.start_point, start_direction)
            evaluator.reset_score()
        except Exception:
            pass

pyglet.clock.schedule_interval(game_loop, 1/60.0)

@window.event
def on_draw():
    window.clear()
    try:
        drawer.draw()
    except Exception as e:
        log.exception("Drawer.draw() failed: %s", e)
    try:
        evaluator.draw_lines()
    except Exception as e:
        log.exception("evaluator.draw_lines() failed: %s", e)

@window.event
def on_resize(width, height):
    try:
        drawer.resize_canvas(width, height)
        drawer.draw()
    except Exception as e:
        log.exception("Error handling resize: %s", e)

if __name__ == "__main__":
    log.info("Starting game window (size %dx%d).", display_w, display_h)
    try:
        pyglet.app.run()
    except Exception as e:
        log.exception("Pyglet app terminated with exception: %s", e)
