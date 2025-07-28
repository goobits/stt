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
    # Try to import app_hooks from same directory as CLI
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

# Built-in commands

def builtin_upgrade_command(check_only=False, pre=False, version=None, dry_run=False):
    """Built-in upgrade function for Speech-to-Text CLI - uses enhanced setup.sh script."""
    import subprocess
    import sys
    from pathlib import Path

    if check_only:
        print(f"Checking for updates to Speech-to-Text CLI...")
        print("Update check not yet implemented. Run without --check to upgrade.")
        return

    if dry_run:
        print("Dry run - would execute: pipx upgrade goobits-stt")
        return

    # Find the setup.sh script - look in common locations
    setup_script = None
    search_paths = [
        Path.cwd() / "setup.sh",  # Current directory
        Path(__file__).parent.parent / "setup.sh",  # Package directory
        Path.home() / ".local" / "share" / "goobits-stt" / "setup.sh",  # User data
    ]
    
    for path in search_paths:
        if path.exists():
            setup_script = path
            break
    
    if setup_script is None:
        # Fallback to basic upgrade if setup.sh not found
        print(f"Enhanced setup script not found. Using basic upgrade for Speech-to-Text CLI...")
        import shutil
        
        package_name = "goobits-stt"
        pypi_name = "goobits-stt"
        
        if shutil.which("pipx"):
            result = subprocess.run(["pipx", "list"], capture_output=True, text=True)
            if package_name in result.stdout or pypi_name in result.stdout:
                cmd = ["pipx", "upgrade", pypi_name]
            else:
                cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pypi_name]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pypi_name]
        
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"✅ Speech-to-Text CLI upgraded successfully!")
            print(f"Run 'stt --version' to verify the new version.")
        else:
            print(f"❌ Upgrade failed with exit code {result.returncode}")
            sys.exit(1)
        return

    # Use the enhanced setup.sh script
    result = subprocess.run([str(setup_script), "upgrade"])
    sys.exit(result.returncode)


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
                
            # Skip core system files that aren't plugins
            if plugin_file.name in ["loader.py", "__init__.py"]:
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




@click.option("--model",
    type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']),
    default="base",
    help="Whisper model size to use"
)

@click.option("--language",
    type=str,
    default="en",
    help="Language code for transcription"
)

@click.option("--device",
    type=click.Choice(['cpu', 'cuda', 'mps']),
    default="cpu",
    help="Compute device to use"
)

@click.option("--no-formatting",
    is_flag=True,
    help="Disable text formatting and punctuation"
)

@click.option("--json",
    is_flag=True,
    help="Output results in JSON format"
)

@click.option("--debug",
    is_flag=True,
    help="Enable debug logging"
)

@click.option("--config",
    type=str,
    help="Path to configuration file"
)

@click.option("--sample-rate",
    type=int,
    default=16000,
    help="Audio sample rate in Hz"
)


def main(ctx, model=None, language=None, device=None, no_formatting=False, json=False, debug=False, config=None, sample_rate=None):
    """[bold color(6)]stt v1.0.0[/bold color(6)] - Real-time speech-to-text transcription

    

    
    \b
    """

    
    
    # Store global options in context for use by commands
    
    if ctx.obj is None:
        ctx.obj = {}
    
    ctx.obj['model'] = model
    
    ctx.obj['language'] = language
    
    ctx.obj['device'] = device
    
    ctx.obj['no-formatting'] = no_formatting
    
    ctx.obj['json'] = json
    
    ctx.obj['debug'] = debug
    
    ctx.obj['config'] = config
    
    ctx.obj['sample-rate'] = sample_rate
    
    

    pass



# Built-in upgrade command (enabled by default)

@main.command()
@click.option('--check', is_flag=True, help='Check for updates without installing')
@click.option('--version', type=str, help='Install specific version')
@click.option('--pre', is_flag=True, help='Include pre-release versions')
@click.option('--dry-run', is_flag=True, help='Show what would be done without doing it')
def upgrade(check, version, pre, dry_run):
    """Upgrade Speech-to-Text CLI to the latest version."""
    builtin_upgrade_command(check_only=check, version=version, pre=pre, dry_run=dry_run)




@main.command()
@click.pass_context


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

def listen(ctx, device, language, model, hold_to_talk, json, debug, config, no_formatting, sample_rate):
    """Record once and transcribe"""
    
    # Check for built-in commands first
    
    # Standard command - use the existing hook pattern
    hook_name = f"on_listen"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'listen'  # Pass command name for all commands
        
        
        
        
        
        
        kwargs['device'] = device
        
        
        
        
        kwargs['language'] = language
        
        
        
        
        kwargs['model'] = model
        
        
        
        
        kwargs['hold_to_talk'] = hold_to_talk
        
        
        
        
        kwargs['json'] = json
        
        
        
        
        kwargs['debug'] = debug
        
        
        
        
        kwargs['config'] = config
        
        
        
        
        kwargs['no_formatting'] = no_formatting
        
        
        
        
        kwargs['sample_rate'] = sample_rate
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
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
@click.pass_context


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

def live(ctx, device, language, model, tap_to_talk, json, debug, config, no_formatting, sample_rate):
    """Live conversation mode"""
    
    # Check for built-in commands first
    
    # Standard command - use the existing hook pattern
    hook_name = f"on_live"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'live'  # Pass command name for all commands
        
        
        
        
        
        
        kwargs['device'] = device
        
        
        
        
        kwargs['language'] = language
        
        
        
        
        kwargs['model'] = model
        
        
        
        
        kwargs['tap_to_talk'] = tap_to_talk
        
        
        
        
        kwargs['json'] = json
        
        
        
        
        kwargs['debug'] = debug
        
        
        
        
        kwargs['config'] = config
        
        
        
        
        kwargs['no_formatting'] = no_formatting
        
        
        
        
        kwargs['sample_rate'] = sample_rate
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
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
@click.pass_context


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

def serve(ctx, port, host, debug, config):
    """Start transcription server"""
    
    # Check for built-in commands first
    
    # Standard command - use the existing hook pattern
    hook_name = f"on_serve"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'serve'  # Pass command name for all commands
        
        
        
        
        
        
        kwargs['port'] = port
        
        
        
        
        kwargs['host'] = host
        
        
        
        
        kwargs['debug'] = debug
        
        
        
        
        kwargs['config'] = config
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing serve command...")
        
        
        
        click.echo(f"  port: {port}")
        
        click.echo(f"  host: {host}")
        
        click.echo(f"  debug: {debug}")
        
        click.echo(f"  config: {config}")
        
        
    
    




@main.command()
@click.pass_context


def status(ctx):
    """Show system status and capabilities"""
    
    # Check for built-in commands first
    
    # Standard command - use the existing hook pattern
    hook_name = f"on_status"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'status'  # Pass command name for all commands
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing status command...")
        
        
    
    




@main.command()
@click.pass_context


def models(ctx):
    """List available Whisper models"""
    
    # Check for built-in commands first
    
    # Standard command - use the existing hook pattern
    hook_name = f"on_models"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'models'  # Pass command name for all commands
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing models command...")
        
        
    
    




@main.group()
def config():
    """Manage STT configuration"""
    pass


@config.command()
@click.pass_context


def list(ctx):
    """List all configuration settings"""
    # Check if hook function exists
    hook_name = f"on_config_list"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'list'  # Pass command name for all commands
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing list command...")
        
        

@config.command()
@click.pass_context

@click.argument(
    "KEY"
)


def get(ctx, key):
    """Get specific configuration value"""
    # Check if hook function exists
    hook_name = f"on_config_get"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'get'  # Pass command name for all commands
        
        
        kwargs['key'] = key
        
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
        return result
    else:
        # Default placeholder behavior
        click.echo(f"Executing get command...")
        
        
        click.echo(f"  key: {key}")
        
        
        

@config.command()
@click.pass_context

@click.argument(
    "KEY"
)

@click.argument(
    "VALUE"
)


def set(ctx, key, value):
    """Set configuration value"""
    # Check if hook function exists
    hook_name = f"on_config_set"
    if app_hooks and hasattr(app_hooks, hook_name):
        # Call the hook with all parameters
        hook_func = getattr(app_hooks, hook_name)
        
        # Prepare arguments including global options
        kwargs = {}
        kwargs['command_name'] = 'set'  # Pass command name for all commands
        
        
        kwargs['key'] = key
        
        kwargs['value'] = value
        
        
        
        
        # Add global options from context
        
        if ctx.obj:
            
            kwargs['model'] = ctx.obj.get('model')
            
            kwargs['language'] = ctx.obj.get('language')
            
            kwargs['device'] = ctx.obj.get('device')
            
            kwargs['no_formatting'] = ctx.obj.get('no-formatting')
            
            kwargs['json'] = ctx.obj.get('json')
            
            kwargs['debug'] = ctx.obj.get('debug')
            
            kwargs['config'] = ctx.obj.get('config')
            
            kwargs['sample_rate'] = ctx.obj.get('sample-rate')
            
        
        
        result = hook_func(**kwargs)
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