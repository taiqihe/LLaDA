const { test, expect } = require('@playwright/test');

test.describe('LLaDA Visualizer Model Workflow', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();

    // Set up console logging for debugging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error('Browser console error:', msg.text());
      }
    });

    // Handle WebSocket errors gracefully
    page.on('pageerror', error => {
      console.error('Page error:', error.message);
    });

    await page.goto('/');

    // Wait for the page to be fully loaded
    await expect(page.locator('h1')).toContainText('Diffusion Language Model Visualizer');

    // Wait for WebSocket connection
    await expect(page.locator('#status')).toContainText('Connected to server', { timeout: 10000 });
  });

  test('should load model successfully', async () => {
    // Set the model path to the specified local model
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');

    // Click load model button
    await page.click('#load_model');

    // Button should be disabled while loading
    await expect(page.locator('#load_model')).toBeDisabled();

    // Wait for model to load (this may take a while)
    await expect(page.locator('#status')).toContainText('Model loaded successfully', {
      timeout: 180000 // 3 minutes for model loading
    });

    // Model status should be updated
    await expect(page.locator('#model_status')).toBeVisible();
    await expect(page.locator('#model_info')).toContainText('/Users/theo/Projects/models/LLaDA-8B-Base');

    // Initialize button should be enabled
    await expect(page.locator('#initialize')).toBeEnabled();

    // Load model button should be re-enabled
    await expect(page.locator('#load_model')).toBeEnabled();
  });

  test('should initialize generation successfully', async () => {
    // First load the model
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    // Set a prompt
    const testPrompt = 'The capital of France is';
    await page.fill('#prompt', testPrompt);

    // Set generation parameters
    await page.fill('#gen_length', '64');

    // Click initialize
    await page.click('#initialize');

    // Wait for initialization to complete
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    // Visualization should be visible
    await expect(page.locator('#visualization')).toBeVisible();

    // Step info should be displayed
    await expect(page.locator('#current_step')).toBeVisible();
    await expect(page.locator('#current_block')).toBeVisible();

    // Control buttons should be enabled
    await expect(page.locator('#forward_pass')).toBeEnabled();

    // Token grid should show tokens
    const tokenRows = page.locator('#token_grid .token-row');
    await expect(tokenRows.first()).toBeVisible();

    // Check that prompt tokens are marked correctly
    const promptTokens = page.locator('#token_grid .token-row.prompt');
    await expect(promptTokens.first()).toBeVisible();
  });

  test('should run forward pass successfully', async () => {
    // Load model and initialize
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    const testPrompt = 'The capital of France is';
    await page.fill('#prompt', testPrompt);
    await page.fill('#gen_length', '32');

    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    // Run forward pass
    await page.click('#forward_pass');

    // Wait for forward pass to complete
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Visualization should enter forward pass mode
    await expect(page.locator('#visualization')).toHaveClass(/forward-pass-mode/);

    // Probability processing section should be enabled
    await expect(page.locator('#probability_processing')).not.toHaveClass(/disabled/);

    // Token selection section should be enabled
    await expect(page.locator('#token_selection')).not.toHaveClass(/disabled/);

    // Apply probability settings button should be enabled
    await expect(page.locator('#apply_probability_settings')).toBeEnabled();

    // Token candidates should be visible
    await expect(page.locator('.token-candidate').first()).toBeVisible();

    // Each token candidate should show probability and logit values
    const firstCandidate = page.locator('.token-candidate').first();
    await expect(firstCandidate.locator('.token-prob')).toBeVisible();
    await expect(firstCandidate.locator('.token-logit')).toBeVisible();
  });

  test('should show loading indicator during forward pass', async () => {
    // Load model and initialize
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test loading indicator');
    await page.fill('#gen_length', '32');

    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    // Before clicking forward pass, the button text should be visible
    await expect(page.locator('#forward_pass_text')).toBeVisible();
    await expect(page.locator('#forward_pass_loader')).toBeHidden();

    // Click forward pass
    const forwardPassButton = page.locator('#forward_pass');
    await forwardPassButton.click();

    // Button should be disabled during execution
    await expect(forwardPassButton).toBeDisabled();

    // Wait for forward pass to complete
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // After completion, button should be re-enabled
    await expect(forwardPassButton).toBeEnabled();
    await expect(page.locator('#forward_pass_text')).toBeVisible();
    await expect(page.locator('#forward_pass_loader')).toBeHidden();
  });

  test('should handle probability settings adjustments', async () => {
    // Complete the setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Hello world');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Adjust temperature settings
    await page.fill('#softmax_temperature', '0.8');
    await page.fill('#visual_top_k', '25');
    await page.fill('#actual_top_k', '15');
    await page.fill('#top_p', '0.9');

    // Enable Gumbel noise
    await page.check('#apply_gumbel_noise');
    await expect(page.locator('#gumbel_temperature')).toBeEnabled();
    await page.fill('#gumbel_temperature', '0.5');

    // Apply probability settings
    await page.click('#apply_probability_settings');

    // Wait for reprocessing to complete
    await expect(page.locator('#status')).toContainText('Probability settings applied successfully', { timeout: 60000 });

    // Verify the probabilities have been updated
    await expect(page.locator('.token-candidate').first()).toBeVisible();
  });

  test('should allow manual token selection', async () => {
    // Setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test prompt');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Find a token candidate and click it
    const firstCandidate = page.locator('.token-candidate').first();
    await firstCandidate.click();

    // The candidate should be marked as selected
    await expect(firstCandidate).toHaveClass(/selected/);

    // Manual selections panel should be visible
    await expect(page.locator('#manual_selections')).toBeVisible();
    await expect(page.locator('#selections_list .selection-item')).toHaveCount(1);

    // Click the same token again to deselect
    await firstCandidate.click();
    await expect(firstCandidate).not.toHaveClass(/selected/);

    // Manual selections should be hidden again
    await expect(page.locator('#manual_selections')).toBeHidden();
  });

  test('should handle WebSocket connection errors gracefully', async () => {
    // Wait for initial connection
    await expect(page.locator('#status')).toContainText('Connected to server');

    // Simulate trying to load a non-existent model
    await page.fill('#model_path', '/non/existent/model/path');
    await page.click('#load_model');

    // Should show an error
    await expect(page.locator('#status')).toContainText('Failed to load', { timeout: 30000 });
    await expect(page.locator('#model_status')).toContainText('Failed to load');

    // Load button should be re-enabled
    await expect(page.locator('#load_model')).toBeEnabled();
  });

  test('should validate required fields', async () => {
    // Try to initialize without loading a model first
    await page.fill('#prompt', 'Test prompt');

    // Initialize button should be disabled
    await expect(page.locator('#initialize')).toBeDisabled();

    // Load a model first
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    // Now initialize should be enabled
    await expect(page.locator('#initialize')).toBeEnabled();

    // Clear the prompt and try to initialize
    await page.fill('#prompt', '');
    await page.click('#initialize');

    // Should show validation error
    await expect(page.locator('#status')).toContainText('Please enter a prompt');
  });

  test.afterEach(async () => {
    await page.close();
  });
});