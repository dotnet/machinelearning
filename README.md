

# Machine Learning for .NET

[ML.NET](https://www.microsoft.com/net/learn/apps/machine-learning-and-ai/ml-dotnet) is a cross-platform open-source machine learning framework which makes machine learning accessible to .NET developers.

ML.NET allows .NET developers to develop their own models and infuse custom machine learning into their applications, using .NET, even without prior expertise in developing or tuning machine learning models.

ML.NET was originally developed in Microsoft Research, and evolved into a significant framework over the last decade and is used across many product groups in Microsoft like Windows, Bing, PowerPoint, Excel and more.

ML.NET enables machine learning tasks like classification (for example: support text classification, sentiment analysis) and regression (for example, price-prediction).

Along with these ML capabilities, this first release of ML.NET also brings the first draft of .NET APIs for training models, using models for predictions, as well as the core components of this framework such as learning algorithms, transforms, and ML data structures. 

## Installation
https://github.com/oscarg933/official-images.git
[![NuGet Status](https://img.shields.io/nuget/v/Microsoft.ML.svg?style=flat)](https://www.nuget.org/packages/Microsoft.ML/)

ML.NET runs on Windows, Linux, and macOS - any platform where x64 [.NET Core](https://github.com/dotnet/core) or later is available. In addition, .NET Framework on Windows x64 is also supported.

The current release is 0.6. Check out the [release notes](docs/release-notes/0.6/release-0.6.md) to see what's new.

First, ensure you have installed [.NET Core 2.0](https://www.microsoft.com/net/learn/get-started) or later. ML.NET also works on the .NET Framework. Note that ML.NET currently must run in a 64-bit process.

Once you have an app, you can install the ML.NET NuGet package from the .NET Core CLI using:
```
dotnet add package Microsoft.ML
```

or from the NuGet package manager:
```
Install-Package Microsoft.ML
```

Or alternatively, you can add the Microsoft.ML package from within Visual Studio's NuGet package manager or via [Paket](https://github.com/fsprojects/Paket).

Daily NuGet builds of the project are also available in our [MyGet](https://dotnet.myget.org/feed/dotnet-core/package/nuget/Microsoft.ML) feed:

> [https://dotnet.myget.org/F/dotnet-core/api/v3/index.json](https://dotnet.myget.org/F/dotnet-core/api/v3/index.json)

## Building

To build ML.NET from source please visit our [developers guide](docs/project-docs/developer-guide.md).

|    | x64 Debug | x64 Release |
|:---|----------------:|------------------:|
|**Linux**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**macOS**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|
|**Windows**|[![x64-debug](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|[![x64-release](https://dnceng.visualstudio.com/public/_apis/build/status/dotnet/machinelearning/MachineLearning-CI?branchName=master)](https://dnceng.visualstudio.com/DotNet-Public/_build/latest?definitionId=104&branch=master)|

## Contributing

We welcome contributions! Please review our [contribution guide](CONTRIBUTING.md).

## Community

Please join our community on Gitter [![Join the chat at https://gitter.im/dotnet/mlnet](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/dotnet/mlnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This project has adopted the code of conduct defined by the [Contributor Covenant](https://contributor-covenant.org/) to clarify expected behavior in our community.
For more information, see the [.NET Foundation Code of Conduct](https://dotnetfoundation.org/code-of-conduct).

## Examples

Here's an example of code to train a model to predict sentiment from text samples. 
(You can find a sample of the legacy API [here](test/Microsoft.ML.Tests/Scenarios/SentimentPredictionTests.cs)):

```C#
var env = new LocalEnvironment();
var reader = TextLoader.CreateReader(env, ctx => (
        Target: ctx.LoadFloat(2),
        FeatureVector: ctx.LoadFloat(3, 6)),
        separator: ',',
        hasHeader: true);
var data = reader.Read(new MultiFileSource(dataPath));
var classification = new MulticlassClassificationContext(env);
var learningPipeline = reader.MakeNewEstimator()
    .Append(r => (
    r.Target,
    Prediction: classification.Trainers.Sdca(r.Target.ToKey(), r.FeatureVector)));
var model = learningPipeline.Fit(data);

```

Now from the model we can make inferences (predictions):

```C#
var predictionFunc = model.MakePredictionFunction<SentimentInput, SentimentPrediction>(env);
var prediction = predictionFunc.Predict(new SentimentData
{
    SentimentText = "Today is a great day!"
});
Console.WriteLine("prediction: " + prediction.Sentiment);
```
A cookbook that shows how to use these APIs for a variety of existing and new scenarios can be found [here](docs/code/MlNetCookBook.md).


## Samples

We have a [repo of samples](https://github.com/dotnet/machinelearning-samples) that you can look at.

## License

ML.NET is licensed under the [MIT license](LICENSE).

## .NET Foundation

ML.NET is a [.NET Foundation](https://www.dotnetfoundation.org/projects) project.

There are many .NET related projects on GitHub.

- [.NET home repo](https://github.com/Microsoft/dotnet) - links to 100s of .NET projects, from Microsoft and the community.


Contents
Selected Version
Azure Bot Service
Request payment
12/12/2017
9 minutes to read
Contributors
Duc Cash Vo  Kamran Iqbal  CathyQian  Robert Standefer  Kim Brandl - MSFT all
In this article
Prerequisites
Payment process overview
Payment Bot sample
Requesting payment
User experience
Processing callbacks
Testing a payment bot
Additional resources
 Note
This topic applies to SDK v3 release. You can find the documentation for the latest version of the SDK v4 here.

If your bot enables users to purchase items, it can request payment by including a special type of button within a rich card. This article describes how to send a payment request using the Bot Builder SDK for Node.js.
Prerequisites
Before you can send a payment request using the Bot Builder SDK for Node.js, you must complete these prerequisite tasks.
Register and configure your bot

Update your bot's environment variables for MicrosoftAppId and MicrosoftAppPassword to the app ID and password values that were generated for your bot during the registration process.
 Note
To find your bot's AppID and AppPassword, see MicrosoftAppID and MicrosoftAppPassword.
Create and configure merchant account

Create and activate a Stripe account if you don't have one already.
Sign in to Seller Center with your Microsoft account.
Within Seller Center, connect your account with Stripe.
Within Seller Center, navigate to the Dashboard and copy the value of MerchantID.
Update the PAYMENTS_MERCHANT_ID environment variable to the value that you copied from the Seller Center Dashboard.
Payment process overview
The payment process comprises three distinct parts:
The bot sends a payment request.
The user signs in with a Microsoft account to provide payment, shipping, and contact information. Callbacks are sent to the bot to indicate when the bot needs to perform certain operations (update shipping address, update shipping option, complete payment).
The bot processes the callbacks that it receives, including shipping address update, shipping option update, and payment complete.
Your bot must implement only step one and step three of this process; step two takes place outside the context of your bot.
Payment Bot sample
The Payment Bot sample provides an example of a bot that sends a payment request by using Node.js. To see this sample bot in action, you can try it out in web chat, add it as a Skype contact, or download the payment bot sample and run it locally using the Bot Framework Emulator.
 Note
To complete the end-to-end payment process using the Payment Bot sample in web chat or Skype, you must specify a valid credit card or debit card within your Microsoft account (i.e., a valid card from a U.S. card issuer). Your card will not be charged and the card's CVV will not be verified, because the Payment Bot sample runs in test mode (i.e., PAYMENTS_LIVEMODE is set to false in .env).
The next few sections of this article describe the three parts of the payment process, in the context of the Payment Bot sample.
Requesting payment
Your bot can request payment from a user by sending a message that contains a rich card with a button that specifies type of "payment". This code snippet from the Payment Bot sample creates a message that contains a Hero card with a Buy button that the user can click (or tap) to initiate the payment process.
JavaScript

Copy
var bot = new builder.UniversalBot(connector, (session) => {

  catalog.getPromotedItem().then(product => {

    // Store userId for later, when reading relatedTo to resume dialog with the receipt.
    var cartId = product.id;
    session.conversationData[CartIdKey] = cartId;
    session.conversationData[cartId] = session.message.address.user.id;

    // Create PaymentRequest obj based on product information.
    var paymentRequest = createPaymentRequest(cartId, product);

    var buyCard = new builder.HeroCard(session)
      .title(product.name)
      .subtitle(util.format('%s %s', product.currency, product.price))
      .text(product.description)
      .images([
        new builder.CardImage(session).url(product.imageUrl)
      ])
      .buttons([
        new builder.CardAction(session)
          .title('Buy')
          .type(payments.PaymentActionType)
          .value(paymentRequest)
      ]);

    session.send(new builder.Message(session)
      .addAttachment(buyCard));
  });
});
In this example, the button's type is specified as payments.PaymentActionType, which the app defines as "payment". The button's value is populated by the createPaymentRequest function, which returns a PaymentRequest object that contains information about supported payment methods, details, and options. For more information about implementation details, see app.js within the Payment Bot sample.
This screenshot shows the Hero card (with Buy button) that's generated by the code snippet above.
Payments sample bot
 Important
Any user that has access to the Buy button may use it to initiate the payment process. Within the context of a group conversation, it is not possible to designate a button for use by only a specific user.
User experience
When a user clicks the Buy button, he or she is directed to the payment web experience to provide all required payment, shipping, and contact information via their Microsoft account.
Microsoft payment
HTTP callbacks

HTTP callbacks will be sent to your bot to indicate that it should perform certain operations. Each callback will be an event that contains these property values:
Property	Value
type	invoke
name	Indicates the type of operation that the bot should perform (e.g., shipping address update, shipping option update, payment complete).
value	The request payload in JSON format.
relatesTo	Describes the channel and user that are associated with the payment request.
 Note
invoke is a special event type that is reserved for use by the Microsoft Bot Framework. The sender of an invoke event will expect your bot to acknowledge the callback by sending an HTTP response.
Processing callbacks
When your bot receives a callback, it should verify that the information specified in the callback is valid and acknowledge the callback by sending an HTTP response.
Shipping Address Update and Shipping Option Update callbacks

When receiving a Shipping Address Update or a Shipping Option Update callback, your bot will be provided with the current state of the payment details from the client in the event's value property. As a merchant, you should treat these callbacks as static, given input payment details you will calculate some output payment details and fail if the input state provided by the client is invalid for any reason.  If the bot determines the given information is valid as-is, simply send HTTP status code 200 OK along with the unmodified payment details. Alternatively, the bot may send HTTP status code 200 OK along with an updated payment details that should be applied before the order can be processed. In some cases, your bot may determine that the updated information is invalid and the order cannot be processed as-is. For example, the user's shipping address may specify a country to which the product supplier does not ship. In that case, the bot may send HTTP status code 200 OK and a message populating the error property of the payment details object. Sending any HTTP status code in the 400 or 500 range to will result in a generic error for the customer.
Payment Complete callbacks

When receiving a Payment Complete callback, your bot will be provided with a copy of the initial, unmodified payment request as well as the payment response objects in the event's value property. The payment response object will contain the final selections made by the customer along with a payment token. Your bot should take the opportunity to recalculate the final payment request based on the initial payment request and the customer's final selections. Assuming the customer's selections are determined to be valid, the bot should verify the amount and currency in the payment token header to ensure that they match the final payment request. If the bot decides to charge the customer it should only charge the amount in the payment token header as this is the price the customer confirmed. If there is a mismatch between the values that the bot expects and the values that it received in the Payment Complete callback, it can fail the payment request by sending HTTP status code 200 OK along with setting the result field to failure.
In addition to verifying payment details, the bot should also verify that the order can be fulfilled, before it initiates payment processing. For example, it may want to verify that the item(s) being purchased are still available in stock. If the values are correct and your payment processor has successfully charged the payment token, then the bot should respond with HTTP status code 200 OK along with setting the result field to success in order for the payment web experience to display the payment confirmation. The payment token that the bot receives can only be used once, by the merchant that requested it, and must be submitted to Stripe (the only payment processor that the Bot Framework currently supports). Sending any HTTP status code in the 400 or 500 range to will result in a generic error for the customer.
This code snippet from the Payment Bot sample processes the callbacks that the bot receives.
JavaScript

Copy
connector.onInvoke((invoke, callback) => {
  console.log('onInvoke', invoke);

  // This is a temporary workaround for the issue that the channelId for "webchat" is mapped to "directline" in the incoming RelatesTo object
  invoke.relatesTo.channelId = invoke.relatesTo.channelId === 'directline' ? 'webchat' : invoke.relatesTo.channelId;

  var storageCtx = {
    address: invoke.relatesTo,
    persistConversationData: true,
    conversationId: invoke.relatesTo.conversation.id
  };

  connector.getData(storageCtx, (err, data) => {
    var cartId = data.conversationData[CartIdKey];
    if (!invoke.relatesTo.user && cartId) {
      // Bot keeps the userId in context.ConversationData[cartId]
      var userId = data.conversationData[cartId];
      invoke.relatesTo.useAuth = true;
      invoke.relatesTo.user = { id: userId };
    }

    // Continue based on PaymentRequest event.
    var paymentRequest = null;
    switch (invoke.name) {
      case payments.Operations.UpdateShippingAddressOperation:
      case payments.Operations.UpdateShippingOptionOperation:
        paymentRequest = invoke.value;

        // Validate address AND shipping method (if selected).
        checkout
          .validateAndCalculateDetails(paymentRequest, paymentRequest.shippingAddress, paymentRequest.shippingOption)
          .then(updatedPaymentRequest => {
            // Return new paymentRequest with updated details.
            callback(null, updatedPaymentRequest, 200);
          }).catch(err => {
            // Return error to onInvoke handler.
            callback(err);
            // Send error message back to user.
            bot.beginDialog(invoke.relatesTo, 'checkout_failed', {
              errorMessage: err.message
            });
          });

        break;

      case payments.Operations.PaymentCompleteOperation:
        var paymentRequestComplete = invoke.value;
        paymentRequest = paymentRequestComplete.paymentRequest;
        var paymentResponse = paymentRequestComplete.paymentResponse;

        // Validate address AND shipping method.
        checkout
          .validateAndCalculateDetails(paymentRequest, paymentResponse.shippingAddress, paymentResponse.shippingOption)
          .then(updatedPaymentRequest =>
            // Process payment.
            checkout
              .processPayment(updatedPaymentRequest, paymentResponse)
              .then(chargeResult => {
                // Return success.
                callback(null, { result: "success" }, 200);
                // Send receipt to user.
                bot.beginDialog(invoke.relatesTo, 'checkout_receipt', {
                  paymentRequest: updatedPaymentRequest,
                  chargeResult: chargeResult
                });
              })
          ).catch(err => {
            // Return error to onInvoke handler.
            callback(err);
            // Send error message back to user.
            bot.beginDialog(invoke.relatesTo, 'checkout_failed', {
              errorMessage: err.message
            });
          });

        break;
    }

  });
});
In this example, the bot examines the name property of the incoming event to identify the type of operation it needs to perform, and then calls the appropriate method(s) to process the callback. For more information about implementation details, see app.js within the Payment Bot sample.
Testing a payment bot
To fully test a bot that requests payment, configure it to run on channels that support Bot Framework payments, like Web Chat and Skype. Alternatively, you can test your bot locally using the Bot Framework Emulator.
 Tip
Callbacks are sent to your bot when a user changes data or clicks Pay during the payment web experience. Therefore, you can test your bot's ability to receive and process callbacks by interacting with the payment web experience yourself.
In the Payment Bot sample, the PAYMENTS_LIVEMODE environment variable in .env determines whether Payment Complete callbacks will contain emulated payment tokens or real payment tokens. If PAYMENTS_LIVEMODE is set to false, a header is added to the bot's outbound payment request to indicate that the bot is in test mode, and the Payment Complete callback will contain an emulated payment token that cannot be charged. If PAYMENTS_LIVEMODE is set to true, the header which indicates that the bot is in test mode is omitted from the bot's outbound payment request, and the Payment Complete callback will contain a real payment token that the bot will submit to Stripe for payment processing. This will be a real transaction that results in charges to the specified payment instrument.
Additional resources
Payment Bot sample
Add rich card attachments to messages
Web Payments at W3C
Feedback

Would you like to provide feedback?

Sign in to give feedback
 
Our new feedback system is built on GitHub Issues. Read about this change in our blog post.
There is currently no feedback for this document. Submitted feedback will appear here.
Feedback
English (United States)
Previous Version Docs Blog Contribute Privacy & Cookies Terms of Use Site Feedback Trademarks
