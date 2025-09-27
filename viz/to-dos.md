
# LLaDA Visualizer Improvement Tasks

## Phase 1: Fix Breaking Changes (High Priority)

### In Progress
- ✅ Remove stepping. The new UI cycle is: initialize (once) -> forward pass -> select tokens (auto or manual) -> apply -> manually forward pass to update logits
- 🔄 Fix websocket_handlers.py broken imports and method calls to use TokenTracker
- 🔄 Clean up models.py commented code and ensure consistent data structures
- 🔄 Test basic server functionality after breaking changes

### Not Started
- Fix probability_processor.py integration with new TokenTracker architecture
- Update main.py to properly handle new configuration system
- Verify all import statements across the codebase

## Phase 2: Core Feature Implementation (High Priority)

### Backend Token Management
- Implement missing TokenTracker.auto_select() method
- Implement missing TokenTracker.apply_selection() method
- Add comprehensive input validation for all WebSocket message parameters
- cache probabilities in the backend so we don't have to pass them from the client to the backend
- apply 2 token number restrictions per position, visual and actual
    - for actual top k, we only keep track of tokens in top k and top p, whichever is the lowest

### Error Handling & Logging
- Add comprehensive error handling throughout the codebase
- Implement proper logging for debugging and monitoring
- Add graceful degradation when model operations fail
- Implement proper cleanup on WebSocket disconnection

## Phase 3: Testing & Documentation (Medium Priority)

### Documentation Updates
- Update API.md to match new backend architecture and message types
- Update ARCHITECTURE.md to reflect TokenTracker changes and new data flow
- Update README.md with current project structure and usage

### Testing Infrastructure
- Fix/update Playwright tests to work with new API
- Add unit tests for TokenTracker class
- Add unit tests for DiffusionModel class
- Add integration tests for WebSocket message flow
- Verify end-to-end functionality with realistic model loading

## Phase 4: Frontend & UI Improvements (Medium Priority)

### User Interface
- add UI for dynamic token selection methods
- auto select should update UI with candidates
- make sure manual token selection update works
- Update frontend JavaScript to use new message types and API endpoints
- Improve error handling and user feedback in the web interface
- Add visual indicators for different token selection strategies

### User Experience
- Add loading states and progress indicators
- Implement proper error messages and recovery suggestions
- Add keyboard shortcuts for common operations
- Improve token visualization and interaction

## Phase 5: Performance & Advanced Features (Low Priority)

### Performance Optimization
- Optimize WebSocket message handling and reduce latency
- Add performance monitoring and metrics collection
- Implement connection pooling and resource management
- Add memory usage optimization for large models

### Advanced Features
- Add support for multiple concurrent generation sessions
- Implement generation branching and comparison
- Add export functionality for generation results
- Add configuration presets and user settings persistence

### Developer Experience
- Add comprehensive API examples and tutorials
- Implement hot-reloading for development
- Add debugging tools and introspection capabilities
- Improve development setup documentation

## Completed Tasks
- ✅ Major architecture refactoring: GenerationEngine → TokenTracker
- ✅ Configuration system consolidation in config.py
- ✅ Model operations cleanup in diffusion_model.py
- ✅ Basic token selection strategy implementation