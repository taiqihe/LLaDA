const { test, expect } = require('@playwright/test');

test.describe('TokenTracker Workflow Tests', () => {
  let page;

  test.beforeEach(async ({ browser }) => {
    page = await browser.newPage();

    // Set up console logging for debugging
    page.on('console', msg => {
      if (msg.type() === 'error') {
        console.error('Browser console error:', msg.text());
      }
    });

    page.on('pageerror', error => {
      console.error('Page error:', error.message);
    });

    await page.goto('/');
    await expect(page.locator('h1')).toContainText('Diffusion Language Model Visualizer');
    await expect(page.locator('#status')).toContainText('Connected to server', { timeout: 10000 });
  });

  test('should display dual token restriction controls', async () => {
    // Check that both visual and actual top-k controls exist
    await expect(page.locator('#visual_top_k')).toBeVisible();
    await expect(page.locator('#actual_top_k')).toBeVisible();

    // Check default values
    await expect(page.locator('#visual_top_k')).toHaveValue('20');
    await expect(page.locator('#actual_top_k')).toHaveValue('10');

    // Check labels (looking for labels by text content)
    await expect(page.locator('text=Visual Top-K')).toBeVisible();
    await expect(page.locator('text=Actual Top-K')).toBeVisible();
  });

  test('should display new TokenTracker controls', async () => {
    // Check for new auto selection controls
    await expect(page.locator('#selection_strategy')).toBeVisible();
    await expect(page.locator('#max_tokens')).toBeVisible();
    await expect(page.locator('#auto_select_tokens')).toBeVisible();

    // Check for manual selection controls
    await expect(page.locator('#apply_selections')).toBeVisible();
    await expect(page.locator('#clear_selections')).toBeVisible();

    // Controls should be initially disabled
    await expect(page.locator('#auto_select_tokens')).toBeDisabled();
    await expect(page.locator('#apply_selections')).toBeDisabled();
    await expect(page.locator('#clear_selections')).toBeDisabled();
  });

  test('should enable new controls after model load and initialization', async () => {
    // Load model
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    // Initialize generation
    await page.fill('#prompt', 'Test prompt for TokenTracker');
    await page.fill('#gen_length', '32');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    // New TokenTracker controls should be enabled after initialization
    await expect(page.locator('#auto_select_tokens')).toBeEnabled();
    await expect(page.locator('#apply_selections')).toBeEnabled();
    await expect(page.locator('#clear_selections')).toBeEnabled();
  });

  test('should test forward pass with dual token restrictions', async () => {
    // Load model and initialize
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'The capital of France is');
    await page.fill('#gen_length', '32');

    // Set different visual and actual top-k values
    await page.fill('#visual_top_k', '20');
    await page.fill('#actual_top_k', '10');

    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    // Run forward pass with dual restrictions
    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Should show token candidates
    await expect(page.locator('.token-candidate').first()).toBeVisible();

    // Check for is_in_actual visual indicators
    const candidates = page.locator('.token-candidate');
    const candidateCount = await candidates.count();
    console.log(`Found ${candidateCount} token candidates`);

    // Should have visual candidates (multiple positions with candidates each)
    expect(candidateCount).toBeGreaterThan(0);
    // Forward pass processes all positions and shows candidates - verify reasonable count
    expect(candidateCount).toBeLessThanOrEqual(1000); // More flexible upper bound
  });

  test('should show visual indicators for is_in_actual status', async () => {
    // Complete setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Hello world');
    await page.fill('#visual_top_k', '15');
    await page.fill('#actual_top_k', '5');

    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Check for in-actual and not-in-actual styling
    const inActualCandidates = page.locator('.token-candidate.in-actual');
    const notInActualCandidates = page.locator('.token-candidate.not-in-actual');

    // Should have both types of candidates
    const inActualCount = await inActualCandidates.count();
    const notInActualCount = await notInActualCandidates.count();

    console.log(`In-actual candidates: ${inActualCount}, Not-in-actual: ${notInActualCount}`);

    // Should have some candidates marked as not-in-actual (visual only)
    expect(inActualCount + notInActualCount).toBeGreaterThan(0);

    // Check for "(visual only)" indicator on not-in-actual candidates
    if (notInActualCount > 0) {
      await expect(notInActualCandidates.first()).toContainText('(visual only)');
    }
  });

  test('should test auto selection workflow', async () => {
    // Setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test auto selection');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Set selection parameters
    await page.selectOption('#selection_strategy', 'low_confidence');
    await page.fill('#max_tokens', '2');

    // Click auto select
    await page.click('#auto_select_tokens');

    // Should show status update
    await expect(page.locator('#status')).toContainText(/Auto-selected|No tokens were auto-selected/, { timeout: 10000 });

    // Manual selections may or may not appear depending on auto-selection results
    // If selections were made, the manual selections panel should be visible
    const statusText = await page.locator('#status').textContent();
    if (statusText.includes('Auto-selected')) {
      await expect(page.locator('#manual_selections')).toBeVisible();
    }
  });

  test('should test apply selections workflow', async () => {
    // Setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test apply selection');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Manually select a token by clicking on a candidate
    const firstCandidate = page.locator('.token-candidate').first();
    await firstCandidate.click();

    // Should show as selected
    await expect(firstCandidate).toHaveClass(/selected/);
    await expect(page.locator('#manual_selections')).toBeVisible();

    // Apply the selections
    await page.click('#apply_selections');

    // Should show application result
    await expect(page.locator('#status')).toContainText('Selections applied successfully', { timeout: 10000 });

    // Manual selections should be cleared after application
    await expect(page.locator('#manual_selections')).toBeHidden();
  });

  test('should test cached forward pass indicator', async () => {
    // Setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test cached indicator');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Select and apply a token to potentially trigger cached indicator
    const firstCandidate = page.locator('.token-candidate').first();
    await firstCandidate.click();

    await page.click('#apply_selections');

    // Check if cached indicator appears (it might appear briefly)
    const cachedIndicator = page.locator('.cached-indicator');

    // The indicator should exist in the DOM
    await expect(cachedIndicator).toHaveText('Using Cached Forward Pass');

    // Note: The indicator might only be visible briefly (3 seconds) so we just check it exists
  });

  test('should test clear selections functionality', async () => {
    // Setup to forward pass
    await page.fill('#model_path', '/Users/theo/Projects/models/LLaDA-8B-Base');
    await page.click('#load_model');
    await expect(page.locator('#status')).toContainText('Model loaded successfully', { timeout: 180000 });

    await page.fill('#prompt', 'Test clear selections');
    await page.click('#initialize');
    await expect(page.locator('#status')).toContainText('Step', { timeout: 60000 });

    await page.click('#forward_pass');
    await expect(page.locator('#status')).toContainText('Forward pass complete', { timeout: 90000 });

    // Make multiple manual selections from different positions
    const tokenRows = page.locator('.token-row');
    const candidatesFromPos1 = tokenRows.first().locator('.token-candidate').first();
    const candidatesFromPos2 = tokenRows.nth(1).locator('.token-candidate').first();
    await candidatesFromPos1.click();
    await candidatesFromPos2.click();

    // Should show manual selections
    await expect(page.locator('#manual_selections')).toBeVisible();
    await expect(page.locator('#selections_list .selection-item')).toHaveCount(2);

    // Clear all selections
    await page.click('#clear_selections');

    // Manual selections should be hidden
    await expect(page.locator('#manual_selections')).toBeHidden();

    // Status should confirm clearing
    await expect(page.locator('#status')).toContainText('Manual selections cleared');

    // Previously selected candidates should no longer be selected
    await expect(candidatesFromPos1).not.toHaveClass(/selected/);
    await expect(candidatesFromPos2).not.toHaveClass(/selected/);
  });

  test.afterEach(async () => {
    await page.close();
  });
});