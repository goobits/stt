#!/usr/bin/env python3
"""Auto-generated from goobits.yaml"""
import os
import sys
import importlib.util
from pathlib import Path
import rich_click as click
from rich_click import RichGroup, RichCommand

# Set up rich-click configuration globally
click.rich_click.USE_RICH_MARKUP = True  
click.rich_click.USE_MARKDOWN = False  # Disable markdown to avoid conflicts
click.rich_click.MARKUP_MODE = "rich"

# Environment variables for additional control
os.environ["RICH_CLICK_USE_RICH_MARKUP"] = "1"
os.environ["RICH_CLICK_FORCE_TERMINAL"] = "1"
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "#ff5555"
click.rich_click.ERRORS_SUGGESTION = "Try running the '--help' flag for more information."
click.rich_click.ERRORS_EPILOGUE = "To find out more, visit https://github.com/anthropics/claude-code"
click.rich_click.MAX_WIDTH = 120  # Set reasonable width
click.rich_click.WIDTH = 120  # Set consistent width
click.rich_click.COLOR_SYSTEM = "auto"
click.rich_click.SHOW_SUBCOMMAND_ALIASES = True
click.rich_click.ALIGN_OPTIONS_SWITCHES = True
click.rich_click.STYLE_OPTION = "#ff79c6"      # Dracula Pink - for option flags
click.rich_click.STYLE_SWITCH = "#50fa7b"      # Dracula Green - for switches
click.rich_click.STYLE_METAVAR = "#8BE9FD not bold"   # Light cyan - for argument types (OPTIONS, COMMAND)  
click.rich_click.STYLE_METAVAR_SEPARATOR = "#6272a4"  # Dracula Comment
click.rich_click.STYLE_HEADER_TEXT = "bold yellow"    # Bold yellow - for section headers
click.rich_click.STYLE_EPILOGUE_TEXT = "#6272a4"      # Dracula Comment
click.rich_click.STYLE_FOOTER_TEXT = "#6272a4"        # Dracula Comment
click.rich_click.STYLE_USAGE = "#BD93F9"              # Purple - for "Usage:" line
click.rich_click.STYLE_USAGE_COMMAND = "bold"         # Bold for main command name
click.rich_click.STYLE_DEPRECATED = "#ff5555"         # Dracula Red
click.rich_click.STYLE_HELPTEXT_FIRST_LINE = "#f8f8f2" # Dracula Foreground
click.rich_click.STYLE_HELPTEXT = "#B3B8C0"           # Light gray - for help descriptions
click.rich_click.STYLE_OPTION_DEFAULT = "#ffb86c"     # Dracula Orange
click.rich_click.STYLE_REQUIRED_SHORT = "#ff5555"     # Dracula Red
click.rich_click.STYLE_REQUIRED_LONG = "#ff5555"      # Dracula Red
click.rich_click.STYLE_OPTIONS_PANEL_BORDER = "dim"   # Dim for subtle borders
click.rich_click.STYLE_COMMANDS_PANEL_BORDER = "dim"  # Dim for subtle borders
click.rich_click.STYLE_COMMAND = "#50fa7b"            # Dracula Green - for command names in list
click.rich_click.STYLE_COMMANDS_TABLE_COLUMN_WIDTH_RATIO = (1, 3)  # Command:Description ratio (1/4 : 3/4)



# Hooks system - try to import app_hooks module
app_hooks = None
try:
    # Try to import from the same directory as this script
    script_dir = Path(__file__).parent
    hooks_path = script_dir / "app_hooks.py"
    
    if hooks_path.exists():
        spec = importlib.util.spec_from_file_location("app_hooks", hooks_path)
        app_hooks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_hooks)
    else:
        # Try to import from Python path
        import app_hooks
except (ImportError, FileNotFoundError):
    # No hooks module found, use default behavior
    pass

def load_plugins(cli_group):
    """Load plugins from the conventional plugin directory."""
    # Define plugin directories to search
    plugin_dirs = [
        # User-specific plugin directory
        Path.home() / ".config" / "goobits" / "stt" / "plugins",
        # Local plugin directory (same as script)
        Path(__file__).parent / "plugins",
    ]
    
    for plugin_dir in plugin_dirs:
        if not plugin_dir.exists():
            continue
            
        # Add plugin directory to Python path
        sys.path.insert(0, str(plugin_dir))
        
        # Scan for plugin files
        for plugin_file in plugin_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
                
            plugin_name = plugin_file.stem
            
            try:
                # Import the plugin module
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)
                
                # Call register_plugin if it exists
                if hasattr(plugin_module, "register_plugin"):
                    plugin_module.register_plugin(cli_group)
                    click.echo(f"Loaded plugin: {plugin_name}", err=True)
            except Exception as e:
                click.echo(f"Failed to load plugin {plugin_name}: {e}", err=True)







def get_version():
    """Get version from pyproject.toml or __init__.py"""
    import re
    
    try:
        # Try to get version from pyproject.toml FIRST (most authoritative)
        toml_path = Path(__file__).parent.parent / "pyproject.toml"
        if toml_path.exists():
            content = toml_path.read_text()
            match = re.search(r'version\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    try:
        # Fallback to __init__.py
        init_path = Path(__file__).parent / "__init__.py"
        if init_path.exists():
            content = init_path.read_text()
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return match.group(1)
    except Exception:
        pass
        
    # Final fallback
    return "1.0.0"






  

  

  

  

  

  





@click.group(cls=RichGroup, context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120})

@click.version_option(version=get_version(), prog_name="stt")
@click.pass_context


def main(ctx):
    """[bold color(6)]stt v1.0.0[/bold color(6)] - Real-time speech-to-text transcription

    

    
    \b
    """

    

    pass





@main.command()


@click.option("--device",
    type=str,
    help="Audio input device to use"
)

@click.option("--language",
    type=str,
    help="Language code for transcription"
)

@click.option("--model",
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
    default="base",
    help="Whisper model to use"
)

@click.option("--hold-to-talk",
    type=str,
    help="Hold key to record"
)

@click.option("--json",
    is_flag=True,
    help="Output as JSON"
)

@click.option("--debug",
    is_flag=True,
    help="Enable debug output"
)

@click.option("--config",
    type=str,
    help="Path to config file"
)

@click.option("--no-formatting",
    is_flag=True,
    help="Disable text formatting"
)

@click.option("--sample-rate",
    type=int,
    default=16000,
    help="Audio sample rate"
)

def listen(device, language, model, hold_to_talk, json, debug, config, no_formatting, sample_rate):
    """Record once and transcribe"""
    # Check if hook function exists
    hook_name = f"on_listen"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func(device, language, model, hold_to_talk, json, debug, config, no_formatting, sample_rate)
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing listen command...")
        
        
        
        click.echo(f"  device: {device}")
        
        click.echo(f"  language: {language}")
        
        click.echo(f"  model: {model}")
        
        click.echo(f"  hold-to-talk: {hold_to_talk}")
        
        click.echo(f"  json: {json}")
        
        click.echo(f"  debug: {debug}")
        
        click.echo(f"  config: {config}")
        
        click.echo(f"  no-formatting: {no_formatting}")
        
        click.echo(f"  sample-rate: {sample_rate}")
        
        




@main.command()


@click.option("--device",
    type=str,
    help="Audio input device to use"
)

@click.option("--language",
    type=str,
    help="Language code for transcription"
)

@click.option("--model",
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
    default="base",
    help="Whisper model to use"
)

@click.option("--tap-to-talk",
    type=str,
    help="Key to tap for push-to-talk"
)

@click.option("--json",
    is_flag=True,
    help="Output as JSON"
)

@click.option("--debug",
    is_flag=True,
    help="Enable debug output"
)

@click.option("--config",
    type=str,
    help="Path to config file"
)

@click.option("--no-formatting",
    is_flag=True,
    help="Disable text formatting"
)

@click.option("--sample-rate",
    type=int,
    default=16000,
    help="Audio sample rate"
)

def live(device, language, model, tap_to_talk, json, debug, config, no_formatting, sample_rate):
    """Live conversation mode"""
    # Check if hook function exists
    hook_name = f"on_live"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func(device, language, model, tap_to_talk, json, debug, config, no_formatting, sample_rate)
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing live command...")
        
        
        
        click.echo(f"  device: {device}")
        
        click.echo(f"  language: {language}")
        
        click.echo(f"  model: {model}")
        
        click.echo(f"  tap-to-talk: {tap_to_talk}")
        
        click.echo(f"  json: {json}")
        
        click.echo(f"  debug: {debug}")
        
        click.echo(f"  config: {config}")
        
        click.echo(f"  no-formatting: {no_formatting}")
        
        click.echo(f"  sample-rate: {sample_rate}")
        
        




@main.command()


@click.option("--port",
    type=int,
    default=8769,
    help="Port to run server on"
)

@click.option("--host",
    type=str,
    default="0.0.0.0",
    help="Host to bind to"
)

@click.option("--debug",
    is_flag=True,
    help="Enable debug output"
)

@click.option("--config",
    type=str,
    help="Path to config file"
)

def serve(port, host, debug, config):
    """Start transcription server"""
    # Check if hook function exists
    hook_name = f"on_serve"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func(port, host, debug, config)
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing serve command...")
        
        
        
        click.echo(f"  port: {port}")
        
        click.echo(f"  host: {host}")
        
        click.echo(f"  debug: {debug}")
        
        click.echo(f"  config: {config}")
        
        




@main.command()


def status():
    """Show system status and capabilities"""
    # Check if hook function exists
    hook_name = f"on_status"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func()
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing status command...")
        
        




@main.command()


def models():
    """List available Whisper models"""
    # Check if hook function exists
    hook_name = f"on_models"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func()
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing models command...")
        
        




@main.group()
def config():
    """Manage STT configuration"""
    pass


@config.command()


def list():
    """List all configuration settings"""
    # Check if hook function exists
    hook_name = f"on_config_list"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func()
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing list command...")
        
        

@config.command()

@click.argument(
    "KEY"
)


def get(key):
    """Get specific configuration value"""
    # Check if hook function exists
    hook_name = f"on_config_get"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func(key)
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing get command...")
        
        
        click.echo(f"  key: {key}")
        
        
        

@config.command()

@click.argument(
    "KEY"
)

@click.argument(
    "VALUE"
)


def set(key, value):
    """Set configuration value"""
    # Check if hook function exists
    hook_name = f"on_config_set"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        result = hook_func(key, value)
        
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing set command...")
        
        
        click.echo(f"  key: {key}")
        
        click.echo(f"  value: {value}")
        
        
        





def cli_entry():
    """Entry point for the CLI when installed via pipx."""
    # Load plugins before running the CLI
    load_plugins(main)
    main()

if __name__ == "__main__":
    cli_entry()