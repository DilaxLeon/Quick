/**
 * User Service
 * Handles user-related operations including subscription management
 */

const admin = require('firebase-admin');
const db = admin.firestore();

/**
 * Update a user's subscription details
 * @param {String} userId - Firebase user ID
 * @param {Object} subscriptionData - Subscription data to update
 * @returns {Promise} - Result of the update operation
 */
const updateUserSubscription = async (userId, subscriptionData) => {
  try {
    if (!userId) {
      throw new Error('User ID is required');
    }
    
    const userRef = db.collection('users').doc(userId);
    const userDoc = await userRef.get();
    
    if (!userDoc.exists) {
      throw new Error(`User with ID ${userId} not found`);
    }
    
    const userData = userDoc.data();
    
    // Handle credit updates if specified
    if (subscriptionData.creditsToAdd && typeof subscriptionData.creditsToAdd === 'number') {
      const currentCredits = userData.credits || 0;
      subscriptionData.credits = currentCredits + subscriptionData.creditsToAdd;
      delete subscriptionData.creditsToAdd; // Remove the temporary field
    }
    
    // Update the user document
    await userRef.update({
      ...subscriptionData,
      updatedAt: admin.firestore.FieldValue.serverTimestamp()
    });
    
    console.log(`Updated subscription for user ${userId}`);
    return { success: true };
  } catch (error) {
    console.error('Error updating user subscription:', error);
    return { success: false, error: error.message };
  }
};

/**
 * Get a user's subscription details
 * @param {String} userId - Firebase user ID
 * @returns {Promise} - User subscription data
 */
const getUserSubscription = async (userId) => {
  try {
    if (!userId) {
      throw new Error('User ID is required');
    }
    
    const userRef = db.collection('users').doc(userId);
    const userDoc = await userRef.get();
    
    if (!userDoc.exists) {
      throw new Error(`User with ID ${userId} not found`);
    }
    
    const userData = userDoc.data();
    
    // Extract subscription-related fields
    const subscriptionData = {
      plan: userData.plan || 'free',
      credits: userData.credits || 0,
      subscriptionId: userData.subscriptionId || null,
      subscriptionStatus: userData.subscriptionStatus || null,
      billingCycle: userData.billingCycle || null,
      nextBillingDate: userData.nextBillingDate || null,
      lastPaymentDate: userData.lastPaymentDate || null
    };
    
    return { success: true, data: subscriptionData };
  } catch (error) {
    console.error('Error getting user subscription:', error);
    return { success: false, error: error.message };
  }
};

module.exports = {
  updateUserSubscription,
  getUserSubscription
};